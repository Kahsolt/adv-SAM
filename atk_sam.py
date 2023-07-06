#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/07/05 

import random
from tqdm import tqdm
from traceback import print_exc

import torch
import torch.nn.functional as F
from torch.autograd import grad
from torch import Tensor
from numpy import ndarray
from moviepy.editor import ImageSequenceClip

from run_amg import *
from segment_anything.modeling import Sam
from segment_anything.utils.transforms import ResizeLongestSide

OUT_PATH = Path('out') ; OUT_PATH.mkdir(exist_ok=True)


class SamForwarder():

  def __init__(self, sam:Sam):
    super().__init__()

    self.model = sam
    self.canvas_size = sam.image_encoder.img_size
    self.device = sam.device
    self.transform = ResizeLongestSide(sam.image_encoder.img_size)

  def norm_image(self, x:Tensor) -> Tensor:
    return (x - self.model.pixel_mean) / self.model.pixel_std

  def denorm_image(self, x:Tensor) -> Tensor:
    return x * self.model.pixel_std + self.model.pixel_mean

  def preprocess(self, x:Tensor, is_edge:bool=False) -> Tensor:
    if not is_edge: x = self.norm_image(x)
    h, w = x.shape[-2:]
    return F.pad(x, (0, self.canvas_size - w, 0, self.canvas_size - h))

  def postprocess(self, x:Tensor) -> Tensor:
    x = F.interpolate(x, (self.canvas_size, self.canvas_size), mode="bilinear", align_corners=False)
    x = x[..., :self.input_size[0], :self.input_size[1]]
    return F.interpolate(x, self.original_size, mode='bilinear', align_corners=False)

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

    # zero pad to 1024x1024
    return self.preprocess(X, is_edge)

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

  def forward(self, image:Tensor, point_coords:Tensor=None, point_labels:Tensor=None, boxes:Tensor=None, mask_input:Tensor=None, return_logits:bool=False) -> Tuple[Tensor, Tensor]:
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
    if not return_logits: masks = masks > self.model.mask_threshold

    return masks[0], iou_predictions[0]


@torch.enable_grad()
def pgd_mask(args, forwarder:SamForwarder, img:npimg_u8, edge:npimg_b1=None, prompts:tuple=tuple()) -> npimg_u8:
  if 'get value limits, rescale eps/alpha':
    # NOTE: [C=3, H=1, W=2], the normalized image pixel value is strictly in this value range
    X_limit = forwarder.norm_image(torch.ByteTensor([0, 255]).unsqueeze_(0).unsqueeze_(0).repeat([1, 3, 1, 1]).to(forwarder.device)).squeeze_()
    rng = X_limit.max() - X_limit.min()
    eps   = args.eps   * rng
    alpha = args.alpha * rng
    steps = args.steps

  if 'preprocess raw input':
    X = forwarder.transform_image(img)
    info_t(X, 'X')
    if edge is not None:
      M = forwarder.transform_image(np.tile((np.expand_dims(edge,-1)*255).astype(np.uint8), reps=(1,1,3)), is_edge=True).bool()
      info_t(M, 'M')
    P = forwarder.transform_prompts(*prompts)

  if 'target logits':
    H, W, _ = img.shape
    Y = torch.ones([H, W]).unsqueeze_(0) * args.loss_w
    Y = Y.to(X.device)

  if 'random start AX':
    AX = X.detach().clone()
    noise = torch.empty_like(X).uniform_(-eps, eps)
    if edge is not None: noise = noise * M
    AX = AX + noise

  _cvt = lambda x: x.round().clamp(0, 255).byte().permute(1, 2, 0).detach().cpu().numpy()
  decode_image = lambda x: _cvt(forwarder.postprocess(forwarder.denorm_image(x))[0])
  decode_mask  = lambda x: _cvt(x * 255)

  advs, preds = [], []
  for _ in tqdm(range(steps)):
    AX.requires_grad = True

    pred, iou = forwarder.forward(AX, *P, return_logits=True)
    if 'naive L2 loss':
      loss = F.mse_loss(pred, Y)
    else:
      loss = -iou

    g = grad(loss, AX, loss)[0]
    delta = g.sign() * alpha
    if edge is not None: delta = delta * M

    AX = AX.detach() + delta
    DX = (AX - X).clamp(-eps, eps)
    if edge is not None: DX = DX * M
    AX = (X + DX).detach()
    for i, (vmin, vmax) in enumerate(X_limit):
      AX[:, i, :, :].clamp_(vmin, vmax)

    advs .append(decode_image(AX))
    preds.append(np.tile(decode_mask(pred), reps=(1, 1, 3)))

    print(f'>> iou: {iou.item():.5f}, masked_area: {(pred > 0).sum() / pred.numel():.3%}')

  try:
    ImageSequenceClip(advs,  fps=args.fps).write_videofile(str(OUT_PATH / 'pgd_adv.mp4'))
    ImageSequenceClip(preds, fps=args.fps).write_videofile(str(OUT_PATH / 'pgd_mask.mp4'))
  except:
    print_exc()

  return decode_image(AX)


def fix_prompt_point(coord:str, size:Size) -> Point:
  if coord:
    coord = list(reversed([float(e) for e in coord.split(',')]))
    for i, e in enumerate(coord):
      if e < 1.0: coord[i] = e * size[i]
      coord[i] = int(coord[i])
  else:
    coord = [random.randrange(sz) for sz in size]
  return coord


def run(args):
  # model
  sam = load_sam(args.M)

  # make images
  img = load_img(args.f)  # [H, W, C]
  info_t(img, 'img')
  if args.edge:
    edge = get_mask_edge(img, args.thresh)
    info_t(edge, 'edge')
  else:
    edge = None

  # make prompts
  img_size = img.shape[:-1]
  point = fix_prompt_point(args.point, img_size)
  print(f'>> point: {point}')
  coords = np.expand_dims(np.asarray(point, dtype=np.int32), axis=0)
  labels = np.asarray([1])
  prompts = (coords, labels, None, None)

  # pred X
  predictor = SamPredictor(sam)
  predictor.set_image(img)
  mask, iou, _ = predictor.predict(*prompts, multimask_output=False)
  info_t(mask, 'mask')
  mask = mask[0]  # [C=1, H, W] => [H, W]

  adv = pgd_mask(args, SamForwarder(load_sam(args.M)), img, edge, prompts)
  info_t(adv, 'adv')
  L1: ndarray = np.abs(img.astype(np.int16) - adv.astype(np.int16))
  print('Linf:', L1.max())
  print('L1:', L1.mean())
  diff = minmax_norm(L1)

  # pred AX
  predictor = SamPredictor(sam)
  predictor.set_image(adv)
  mask_adv, iou_adv, _ = predictor.predict(*prompts, multimask_output=False)
  info_t(mask_adv, 'mask_adv')
  mask_adv = mask_adv[0]  # [C=1, H, W] => [H, W]

  if 'show':
    plt.figure(figsize=(12, 8))
    plt.clf()
    plt.subplot(231) ; plt.imshow(img)                   ; plt.title('img')
    plt.subplot(232) ; plt.imshow(mask, cmap='gray')     ; plt.title(f'mask (iou={iou.item():.5f})')
    if edge is not None:
      plt.subplot(233) ; plt.imshow(edge, cmap='gray')     ; plt.title('edge')
    plt.subplot(234) ; plt.imshow(adv)                   ; plt.title('img_adv')
    plt.subplot(235) ; plt.imshow(mask_adv, cmap='gray') ; plt.title(f'mask_adv (iou={iou_adv.item():.5f})')
    plt.subplot(236) ; plt.imshow(diff, cmap='gray')     ; plt.title('diff')
    plt.suptitle(f'point: {point}')
    plt.tight_layout()
    fp = OUT_PATH / 'atk_sam.png'
    plt.savefig(fp, dpi=600)
    print(f'>> savefig to {fp}')
    plt.close()


if __name__ == '__main__':
  parser = get_parser()
  parser.add_argument('--point', help='point coord formatted as h,w; e.g. 0.3,0.4 or 200,300')
  parser.add_argument('--edge',   action='store_true', help='limit PGD to edge')
  parser.add_argument('--thresh', default=0.1,   type=float)
  parser.add_argument('--steps',  default=40,    type=int)
  parser.add_argument('--eps',    default=8/255, type=float)
  parser.add_argument('--alpha',  default=1/255, type=float)
  parser.add_argument('--loss_w', default=-10,   type=float)
  parser.add_argument('--fps',    default=2,     type=int)
  args = get_args(parser)
  
  run(args)
