#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/07/05 

import json
import random
from tqdm import tqdm
from datetime import datetime
from traceback import print_exc

import torch
import torch.nn.functional as F
from torch.autograd import grad
from torch import Tensor
from numpy import ndarray
from moviepy.editor import ImageSequenceClip

from utils import *
from segment_anything.modeling import Sam
from segment_anything.utils.transforms import ResizeLongestSide

OUT_PATH = Path('out') ; OUT_PATH.mkdir(exist_ok=True)
#INTERP_MODE = 'nearest'
INTERP_MODE = 'bilinear'

class SamForwarder:

  def __init__(self, sam:Sam):
    super().__init__()

    self.model = sam
    self.canvas_size = sam.image_encoder.img_size
    self.device = sam.device
    self.transform = ResizeLongestSide(sam.image_encoder.img_size)

  def norm_image(self, x:Tensor) -> Tensor:
    # uint8 [0, 255] -> float32 [-2.xx, 2.xx]
    return (x - self.model.pixel_mean) / self.model.pixel_std

  def denorm_image(self, x:Tensor) -> Tensor:
    return x * self.model.pixel_std + self.model.pixel_mean

  def resize_image(self, x:Tensor) -> Tensor:
    h, w = x.shape[-2:]
    return F.pad(x, (0, self.canvas_size - w, 0, self.canvas_size - h))

  def unresize_image(self, x:Tensor) -> Tensor:
    align_corners = None if INTERP_MODE == 'nearest' else False
    x = F.interpolate(x, (self.canvas_size, self.canvas_size), mode=INTERP_MODE, align_corners=align_corners)
    x = x[..., :self.input_size[0], :self.input_size[1]]
    return F.interpolate(x, self.original_size, mode=INTERP_MODE, align_corners=align_corners)

  def transform_image(self, im:npimg_u8, is_edge:bool=False) -> Tensor:
    # resize along the longest side
    x = self.transform.apply_image(im)                # [H, W, C]
    X: Tensor = torch.from_numpy(x).to(self.device)   # [H, W, C]
    X = X.permute(2, 0, 1).contiguous().unsqueeze_(0) # [B=1, C, H, W]

    assert (
      len(X.shape) == 4 and X.shape[1] == 3 and max(*X.shape[2:]) == self.canvas_size
    ), f"set_torch_image input must be BCHW with long side {self.canvas_size}."

    if not is_edge:
      self.original_size = im.shape[:2]
      self.input_size = tuple(X.shape[-2:])
      X = self.norm_image(X)

    # zero pad to 1024x1024
    return self.resize_image(X)

  def transform_prompts(self, point_coords:ndarray=None, point_labels:ndarray=None, box:ndarray=None, mask_input:ndarray=None) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    coords_torch, labels_torch, box_torch, mask_input_torch = None, None, None, None
    if point_coords is not None:
      assert point_labels is not None, "point_labels must be supplied if point_coords is supplied."
      point_coords = self.transform.apply_coords(point_coords, self.original_size)
      coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
      labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
      coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
    if box is not None:
      box = self.transform.apply_boxes(box, self.original_size)
      box_torch = torch.as_tensor(box, dtype=torch.float, device=self.device)
      box_torch = box_torch[None, :]
    if mask_input is not None:
      mask_input_torch = torch.as_tensor(mask_input, dtype=torch.float, device=self.device)
      mask_input_torch = mask_input_torch[None, :, :, :]
    return coords_torch, labels_torch, box_torch, mask_input_torch

  def forward(self, image:Tensor, point_coords:Tensor=None, point_labels:Tensor=None, boxes:Tensor=None, mask_input:Tensor=None) -> Tuple[Tensor, Tensor]:
    # Embed image
    self.features = self.model.image_encoder(image)
    # Embed prompts
    points = (point_coords, point_labels) if point_coords is not None else None
    sparse_embeddings, dense_embeddings = self.model.prompt_encoder(points, boxes, mask_input)
    # Predict masks
    low_res_masks, iou_predictions = self.model.mask_decoder(
      image_embeddings=self.features,
      image_pe=self.model.prompt_encoder.get_dense_pe(),
      sparse_prompt_embeddings=sparse_embeddings,
      dense_prompt_embeddings=dense_embeddings,
      multimask_output=False,
    )
    # Upscale the masks to the original image resolution
    masks = self.model.postprocess_masks(low_res_masks, self.input_size, self.original_size)

    return masks[0], iou_predictions[0]


@torch.enable_grad()
def pgd(args, sam:Sam, prompts:tuple, img:npimg_u8, tgt:npimg_b1=None, lim:npimg_b1=None) -> npimg_u8:
  forwarder = SamForwarder(sam)
  is_tgt = tgt is not None
  is_lim = lim is not None

  is_01  = lambda x: -1e-5 <= x.min() and x.max() <= 1.0
  npimg_b1_to_u8 = lambda x: np.tile((np.expand_dims(x, -1) * 255).astype(np.uint8), reps=(1, 1, 3))
  denorm         = lambda x: forwarder.denorm_image(x) / 255.0   # normed to [0, 1]
  norm           = lambda x: forwarder.norm_image  (x * 255)     # [0, 1] to normed
  _cvt           = lambda x: torch.as_tensor(x).round().clamp(0, 255).byte().permute(1, 2, 0).detach().cpu().numpy()
  decode_image   = lambda x: _cvt(forwarder.unresize_image(forwarder.denorm_image(x))[0])
  decode_mask    = lambda x: _cvt(x * 255)

  if 'preprocess raw input':
    # NOTE: must call `transform_image` first to set up `self.original_size`
    X = forwarder.transform_image(img)      # [B, C=3, pH, pW], var-mean normed image, ~ [-2.4, 2.4]
    Xo = (denorm(X) * 255).byte().div(255)  # [B, C=3, pH, pW], ~ [0.0, 1.0]
    if is_lim:
      M = forwarder.transform_image(npimg_b1_to_u8(lim), is_edge=True).bool()   # [B, C=3, pH, pW]
    else:
      M = 1.0
    P = forwarder.transform_prompts(*prompts)

  if 'target logits':
    if is_tgt:
      Y = torch.from_numpy(tgt).float()
    else:
      H, W, _ = img.shape
      Y = torch.ones([H, W]) * args.loss_w
    Y = Y.unsqueeze_(0).to(X.device)    # [C=1, oH, oW]

  if 'random start AX':
    AX = X.detach().clone()
    noise = torch.empty_like(X).uniform_(-args.eps, args.eps)
    if is_lim: noise = noise * M
    AX = norm(denorm(AX) + noise)
  
  advs, preds = [], []
  for _ in tqdm(range(args.steps)):
    AX.requires_grad = True

    logits, iou = forwarder.forward(AX, *P)
    pred = logits > forwarder.model.mask_threshold
    if is_tgt:
      loss = F.binary_cross_entropy_with_logits(logits, Y)
    else:
      loss = F.mse_loss(logits, Y)

    g = grad(loss, AX, loss)[0]
    delta = g.sign() * M * args.alpha

    AX = denorm(AX.detach()) - delta
    DX = (AX - Xo).clamp(-args.eps, args.eps)
    AX = norm((Xo + DX).detach().clamp(0.0, 1.0))

    advs .append(decode_image(AX))
    preds.append(np.tile(decode_mask(pred), reps=(1, 1, 3)))

    print(f'>> iou: {iou.item():.5f}, masked_area: {(logits > 0).sum() / logits.numel():.3%}')

  try:
    advs_rep  = advs  + [advs [-1]] * args.fps * 2
    preds_rep = preds + [preds[-1]] * args.fps * 2
    ImageSequenceClip(advs_rep,  fps=args.fps).write_videofile(str(args.log_dp / f'pgd_adv.mp4'))
    ImageSequenceClip(preds_rep, fps=args.fps).write_videofile(str(args.log_dp / f'pgd_mask.mp4'))
  except:
    print_exc()
  
  d: Tensor = torch.abs(Xo - denorm(AX))
  print('Linf (raw):', d.max().item())
  print('L1 (raw):', d.mean().item())

  return decode_image(AX)


def _parse_point(coord:str, size:Size) -> Point:
  if coord:
    point = list(reversed([float(e) for e in coord.split(',')]))
    for i, e in enumerate(point):
      if e < 1.0: point[i] = e * size[i]
      point[i] = int(point[i])
  else:
    point = [random.randrange(sz) for sz in size]
  return point

def make_prompts(point:str, img_size:tuple) -> Tuple:
  point = _parse_point(point, img_size)
  print(f'>> point: {point}')
  coords = np.expand_dims(np.asarray(point, dtype=np.int32), axis=0)
  labels = np.asarray([1])
  return (coords, labels, None, None)
  
def make_pred(sam:Sam, img:npimg_u8, prompts:Tuple) -> Tuple:
  predictor = SamPredictor(sam)
  predictor.set_image(img)
  mask, iou, _ = predictor.predict(*prompts, multimask_output=False)
  mask = mask[0]  # [C=1, H, W] => [H, W]
  return mask, iou

def make_diff(img:npimg_u8, adv:npimg_u8) -> npimg_f32:
  im0 = img / 255.0
  im1 = adv / 255.0
  d: ndarray = np.abs(im0 - im1)
  print('Linf (proc):', d.max())
  print('L1 (proc):', d.mean())
  diff = minmax_norm(d)
  return diff


def run(args):
  # model
  sam = load_sam(args.M)
  # image
  img = load_img(args.f)  # [H, W, C]
  img_size = img.shape[:-1]

  # make edge
  if args.lim:
    edge = get_mask_edge(img, args.thresh)
  else:
    edge = None
  
  # make target
  if args.point_tgt:
    prompts_tgt = make_prompts(args.point_tgt, img_size)
    tgt, _ = make_pred(sam, img, prompts_tgt)
  else:
    tgt = None

  # make prompts
  prompts = make_prompts(args.point, img_size)

  # pred X
  mask, iou = make_pred(sam, img, prompts)

  # attack
  adv = pgd(args, sam, prompts, img, tgt=tgt, lim=edge)
  diff = make_diff(img, adv)

  # pred AX
  mask_adv, iou_adv = make_pred(sam, adv, prompts)

  if 'show':
    plt.figure(figsize=(10, 6))
    plt.clf()
    plt.subplot(231) ; plt.imshow(img)                  ; plt.title('img')
    plt.subplot(232) ; plt.imshow(mask, cmap='gray')    ; plt.title(f'mask (iou={iou.item():.5f})')
    if edge is not None:
      plt.subplot(233) ; plt.imshow(edge, cmap='gray')   ; plt.title('edge')
    plt.subplot(234) ; plt.imshow(adv)                   ; plt.title('img_adv')
    plt.subplot(235) ; plt.imshow(mask_adv, cmap='gray') ; plt.title(f'mask_adv (iou={iou_adv.item():.5f})')
    plt.subplot(236) ; plt.imshow(diff, cmap='gray')     ; plt.title('diff')
    plt.suptitle(f'point: {prompts[0][0]}')
    plt.tight_layout()
    fp = args.log_dp / 'atk_sam.png'
    plt.savefig(fp, dpi=600)
    print(f'>> savefig to {fp}')
    plt.close()


if __name__ == '__main__':
  parser = get_parser()
  parser.add_argument('--point',     help='point coord formatted as h,w; e.g. 0.3,0.4 or 200,300')
  parser.add_argument('--point_tgt', help='alike --point, but specify target mask point, run targetd attack')
  parser.add_argument('--lim',    action='store_true', help='limit PGD to edge area')
  parser.add_argument('--thresh', default=0.1,   type=float, help='edge threshold')
  parser.add_argument('--steps',  default=20,    type=int)
  parser.add_argument('--eps',    default=8/255, type=float)
  parser.add_argument('--alpha',  default=1/255, type=float)
  parser.add_argument('--loss_w', default=-10,   type=float, help='factor for non-targeted attack')
  parser.add_argument('--fps',    default=2,     type=int, help='export video FPS')
  args = get_args(parser)

  def cvt(v:Any) -> Any:
    if isinstance(v, Path): return str(v)
    else: return v

  args.log_dp = OUT_PATH / str(datetime.now().timestamp())
  args.log_dp.mkdir()
  with open(args.log_dp / 'args.json', 'w', encoding='utf-8') as fh:
    json.dump(vars(args), fh, indent=2, default=cvt)

  run(args)
