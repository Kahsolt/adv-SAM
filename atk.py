#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/07/05 

import random
from tqdm import tqdm
from datetime import datetime
from traceback import print_exc

import torch
import torch.nn.functional as F
from torch.autograd import grad
from tqdm import tqdm
try:
  from moviepy.editor import ImageSequenceClip
  HAS_MOVIEPY = True
except ImportError:
  print('>> [warn] missing lib "moviepy", will not generate adv pred step by step')
  HAS_MOVIEPY = False

from utils import *
from utils import get_parser as get_base_parser, get_args as get_base_args
from segment_anything.modeling import Sam
from segment_anything.utils.transforms import ResizeLongestSide

LIM = ['', 'edge', 'smap', 'cam', 'tgt']
CAM_METH = ['GradCAM', 'HiResCAM', 'ScoreCAM', 'GradCAMPlusPlus', 'AblationCAM', 'XGradCAM', 'EigenCAM', 'FullGrad']

class SamForwarder(nn.Module):

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
    INTERP_MODE = 'bilinear'      # 'nearest'
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

  def forward(self, image:Tensor, point_coords:Tensor=None, point_labels:Tensor=None, boxes:Tensor=None, mask_input:Tensor=None, multi_mask:bool=False) -> Tuple[Tensor, Tensor]:
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
      multimask_output=multi_mask,
    )
    # Upscale the masks to the original image resolution
    masks = self.model.postprocess_masks(low_res_masks, self.input_size, self.original_size)

    return masks[0], iou_predictions[0]


@torch.enable_grad()
def pgd(args, fwder:SamForwarder, prompts:Prompts, img:npimg_u8, tgt:npimg_b1=None, lim:npimg_b1=None, 
        multi_mask:bool=False, log:bool=True) -> Union[Tuple[npimg_u8, npimg_b1, float], Tuple[npimg_u8, List[npimg_b1], List[float]]]:

  loss_fn    = F.mse_loss if (tgt is None and not args.force_bce) else F.binary_cross_entropy_with_logits
  is_01      = lambda x: -1e-5 <= x.min() and x.max() <= 1.0
  b1_to_u8   = lambda x: np.tile((np.expand_dims(x, -1) * 255).astype(np.uint8), reps=(1, 1, 3))
  norm       = lambda x: fwder.norm_image  (x  * 255)    # [0, 1] to normed
  denorm     = lambda x: fwder.denorm_image(x) / 255.0   # normed to [0, 1]
  _cvt       = lambda x: torch.as_tensor(x).round().clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
  decode_img = lambda x: _cvt(fwder.unresize_image(fwder.denorm_image(x))[0])
  decode_dx  = lambda x: _cvt(minmax_norm(fwder.unresize_image(x.abs())[0], vmax=args.eps) * 255)
  decode_msk = lambda x: _cvt(x * 255)

  # NOTE: must call `transform_image(img)` first to set up `self.original_size`
  X = fwder.transform_image(img)          # [B, C=3, pH, pW], var-mean normed image, ~ [-2.4, 2.4]
  Xo = (denorm(X) * 255).byte().div(255)  # [B, C=3, pH, pW], ~ [0.0, 1.0]
  M = fwder.transform_image(b1_to_u8(lim), is_edge=True).bool() if lim is not None else 1.0  # [B, C=3, pH, pW]
  P = fwder.transform_prompts(*prompts)
  Y = make_Y(tgt, img, args.loss_w).to(X.device)
  if tgt is None and args.force_bce:
    Y = torch.zeros_like(Y, device=Y.device, dtype=torch.float32)
  Y_bin = torch.zeros_like(Y) if tgt is None else Y

  if 'random start AX':
    AX = X.detach().clone()
    noise = torch.empty_like(X).uniform_(-args.eps, args.eps)
    noise *= M
    AX = norm(denorm(AX) + noise)
  
  is_gen_vid = HAS_MOVIEPY and args.fps > 0
  if is_gen_vid:
    dxs, preds = [], []

  for i in (tqdm if log else enumerate)(range(args.steps)):
    AX.requires_grad = True

    logits, piou = fwder.forward(AX, *P)    # [B=1, H, W], [B=1]
    with torch.no_grad():
      mask = logits > fwder.model.mask_threshold

    if args.meth == 'SegPGD':
      # NOT: THIS OFFSET MAY BE WRONG, BUT KEEP IT AS ORIGINAL IMPL
      lamb = (i - 1) / (args.steps * 2)
      good = mask[0] == Y_bin
      loss_t = F.binary_cross_entropy_with_logits(  good .int() * logits, Y_bin) * lamb
      loss_f = F.binary_cross_entropy_with_logits((~good).int() * logits, Y_bin) * (1 - lamb)
      loss = loss_t + loss_f
    elif args.meth in ['FGSM', 'PGD']:
      loss = loss_fn(logits, Y)
    else:
      raise ValueError(f'unknown loss_fn: {loss_fn}')

    with torch.no_grad():
      if loss.abs() < 1e-5: break     # NOTE: stop early
      g = grad(loss, AX, loss)[0]
      delta = g.sign() * M * args.alpha

      AX = denorm(AX.detach()) - delta
      DX: Tensor = (AX - Xo).clamp(-args.eps, args.eps)
      AX = norm((Xo + DX).detach().clamp(0.0, 1.0))

    if is_gen_vid:
      with torch.no_grad():
        dxs.append(decode_dx(DX))
        preds.append(np.tile(decode_msk(mask), reps=(1, 1, 3)))

    if log:
      print(f'>> loss: {loss.item():.5f}, piou: {piou.item():.5f}, masked_area: {(logits > 0).sum() / logits.numel():.3%}')

  fwder.model.zero_grad()
  gc_everything()

  if is_gen_vid:
    try:
      dxs_rep   = [dxs  [0]] * args.fps * 2 + dxs   + [dxs  [-1]] * args.fps * 4
      preds_rep = [preds[0]] * args.fps * 2 + preds + [preds[-1]] * args.fps * 4
      ImageSequenceClip(dxs_rep,   fps=args.fps).write_videofile(str(args.log_dp / 'pgd_noise.mp4'))
      ImageSequenceClip(preds_rep, fps=args.fps).write_videofile(str(args.log_dp / 'pgd_mask.mp4'))
    except:
      print_exc()
  
  if log:
    with torch.no_grad():
      d: Tensor = torch.abs(Xo - denorm(AX))
    print('Linf (raw):', d.max().item())
    print('L1 (raw):', d.mean().item())
  
  if multi_mask:
    logits, piou = fwder.forward(AX, *P, multi_mask=True)
    mask = logits > fwder.model.mask_threshold
    return decode_img(AX), [mask[i].cpu().numpy() for i in range(len(mask))], piou.tolist()
  else:
    return decode_img(AX), mask[0].cpu().numpy(), piou.item()


def _parse_point(coord:str, size:Size) -> Point:
  if coord:
    point = list(reversed([float(e) for e in coord.split(',')]))
    for i, e in enumerate(point):
      if e < 1.0: point[i] = e * size[i]
      point[i] = int(point[i])
  else:
    point = [random.randrange(sz) for sz in size]
  return point

def make_prompts(point:Union[str, ndarray], img_size:tuple) -> Prompts:
  if isinstance(point, str):
    point = _parse_point(point, img_size)
    print(f'>> point: {point}')
    coords = np.expand_dims(np.asarray(point, dtype=np.int32), axis=0)
  else:
    coords = point
  labels = np.asarray([1], dtype=np.int32)
  return (coords, labels, None, None)

def make_pred(ptor:SamPredictor, img:npimg_u8, prompts:Prompts, multi_mask:bool=False) -> Union[Tuple[npimg_b1, float], Tuple[List[npimg_b1], List[float]]]:
  ptor.set_image(img)
  mask, piou, _ = ptor.predict(*prompts, multimask_output=multi_mask)
  ptor.reset_image()
  if multi_mask:
    return [mask[i] for i in range(len(mask))], piou.tolist()
  else:
    return mask[0], piou.item()    # [C=1, H, W] => [H, W]

def make_tgt(ptor:SamPredictor, img:npimg_u8, point_tgt:Union[str, Point])-> npimg_u8:
  img_size = img.shape[:-1]
  prompts_tgt = make_prompts(point_tgt, img_size)
  tgt, _ = make_pred(ptor, img, prompts_tgt)
  return tgt

def make_Y(tgt:npimg_b1=None, img:npimg_u8=None, w:float=-10.0) -> Tensor:
  if tgt is not None:
    # value in {0, 1}
    Y = torch.from_numpy(tgt).float()
  else:
    # value in {w}, where w is a large negative
    H, W, _ = img.shape
    Y = torch.ones([H, W]) * w
  return Y.unsqueeze_(0)    # [C=1, oH, oW]

@torch.no_grad()
def _make_smap(args, fwder:SamForwarder, prompts:Prompts, img:npimg_u8, tgt:npimg_u8) -> npimg_u8:
  loss_fn = F.mse_loss if tgt is None else F.binary_cross_entropy_with_logits
  decode = lambda x: fwder.unresize_image(x)[0].permute(1, 2, 0).cpu().numpy()

  X = fwder.transform_image(img)
  P = fwder.transform_prompts(*prompts)
  Y = make_Y(tgt, img, args.loss_w).to(X.device)

  with torch.enable_grad():
    X.requires_grad = True
    logits, piou = fwder.forward(X, *P)
    loss = loss_fn(logits, Y)

  g = grad(loss, X, loss)[0]
  psmap = F.tanh(g.abs())
  psmap /= psmap.max()
  smap: npimg_f32 = decode(psmap)    #[H, W, C=3]
  smap = smap.mean(axis=-1)
  lim = smap > args.smap_w

  if args.vis:
    plt.subplot(121) ; plt.title('lim')  ; plt.imshow(lim)
    plt.subplot(122) ; plt.title('smap') ; plt.imshow(smap)
    plt.suptitle(f'smap_w: {args.smap_w}')
    plt.show()

  return lim

@torch.no_grad()
def _make_cam(args, fwder:SamForwarder, prompts:Prompts, img:npimg_u8, tgt:npimg_u8) -> npimg_u8:
  from pytorch_grad_cam.base_cam import BaseCAM
  from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
  from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
  from pytorch_grad_cam.utils.image import show_cam_on_image

  class SAMTarget:
    def __init__(self, mask:Tensor, loss_fn:Callable):
      self.mask = mask    # [H, W]
      self.loss_fn = loss_fn
    def __call__(self, logits:Tensor):
      return self.loss_fn(logits, self.mask)
  
  loss_fn = F.mse_loss if tgt is None else F.binary_cross_entropy_with_logits
  decode = lambda x: fwder.unresize_image(x)[0].permute(1, 2, 0).cpu().numpy()

  X = fwder.transform_image(img)
  P = fwder.transform_prompts(*prompts)
  Y = make_Y(tgt, img, args.loss_w).to(X.device)   # [B=1, H, W]
  L = [fwder.model.mask_decoder.output_upscaling]
  T = [SAMTarget(Y[0], loss_fn)]

  # hijack .forward()
  fwder.forward_orignal = fwder.forward
  fwder.forward = lambda x: fwder.forward_orignal(x, *P)[0]

  cam_peeper: BaseCAM = locals()[args.cam_meth](fwder, L)
  with torch.enable_grad():
    cam = cam_peeper(input_tensor=X, targets=T, eigen_smooth=False)
  del X, P, Y, L, T
  cam_peeper.activations_and_grads: ActivationsAndGradients
  cam_peeper.activations_and_grads.activations.clear()
  cam_peeper.activations_and_grads.gradients  .clear()
  cam_peeper.activations_and_grads.handles    .clear()
  del cam_peeper

  # unhijack .forward()
  fwder.forward = fwder.forward_orignal

  cam = decode(torch.from_numpy(cam).unsqueeze_(1))
  cam = cam.mean(axis=-1)
  lim = cam > args.cam_w
  
  if args.vis:
    #visual = show_cam_on_image(X, cam, use_rgb=True)
    
    plt.subplot(121) ; plt.title('lim') ; plt.imshow(lim)
    plt.subplot(122) ; plt.title('cam') ; plt.imshow(cam)
    plt.suptitle(f'cam_w: {args.cam_w}')
    plt.show()

  return cam

def make_lim(args, img:npimg_u8, tgt:npimg_u8, prompts:Prompts, ptor:SamPredictor, fwder:SamForwarder) -> npimg_b1:
  lim_s: str = args.lim
  inv = False
  if lim_s.startswith('~'):
    lim_s = lim_s[1:]
    inv = True

  if lim_s == 'edge':
    lim = get_edge(img, args.edge_w)
  elif lim_s == 'smap':
    lim = _make_smap(args, fwder, prompts, img, tgt)
  elif lim_s == 'cam':
    lim = _make_cam(args, fwder, prompts, img, tgt)
  elif lim_s == 'tgt':
    lim = ~tgt
  else:   # it should be a point coord
    prompts = make_prompts(lim_s, img.shape[:-1])
    mask, _ = make_pred(ptor, img, prompts)
    lim = mask > ptor.model.mask_threshold

  if inv: lim = ~lim
  return lim


def run(args):
  # model
  sam = load_sam(args.M)
  ptor = SamPredictor(sam)
  fwder = SamForwarder(sam)

  # image
  img = load_img(args.f)  # [H, W, C]
  # make prompts
  prompts = make_prompts(args.point, img.shape[:-1])
  # make tgt
  tgt = make_tgt(ptor, img, args.point_tgt) if args.point_tgt else None
  # make lim
  lim = make_lim(args, img, tgt, prompts, ptor, fwder) if args.lim else None

  # pred X
  mask, piou = make_pred(ptor, img, prompts)
  # attack
  adv, mask_adv, piou_adv = pgd(args, fwder, prompts, img, tgt=tgt, lim=lim)
  # delta
  diff = make_diff(img, adv)
  # pred AX
  if not 'loopback to predictor':
    mask_adv, piou_adv = make_pred(ptor, adv, prompts)

  if 'show':
    cmap = 'gray'
    plt.figure(figsize=(10, 6))
    plt.clf()
    plt.subplot(231) ; plt.imshow(img)            ; plt.title('img')
    plt.subplot(232) ; plt.imshow(mask, cmap)     ; plt.title(f'mask (piou={piou:.5f})')
    if lim is not None:
      plt.subplot(233) ; plt.imshow(lim, cmap)    ; plt.title(f'lim: {args.lim}')
    plt.subplot(234) ; plt.imshow(adv)            ; plt.title('img_adv')
    plt.subplot(235) ; plt.imshow(mask_adv, cmap) ; plt.title(f'mask_adv (piou={piou_adv:.5f})')
    plt.subplot(236) ; plt.imshow(diff, cmap)     ; plt.title('diff (proc)')
    plt.suptitle(f'point: {prompts[0][0]}')
    plt.tight_layout()
    fp = args.log_dp / 'atk_sam.png'
    plt.savefig(fp, dpi=600)
    print(f'>> savefig to {fp}')
    plt.close()


def get_parser() -> ArgumentParser:
  parser = get_base_parser()
  # pgd params
  parser.add_argument('--meth',   default='PGD', choices=['FSGM', 'PGD', 'SegPGD'])
  parser.add_argument('--steps',  default=40,    type=int)
  parser.add_argument('--eps',    default=8/255, type=float)
  parser.add_argument('--alpha',  default=1/255, type=float)
  # non-targeted target
  parser.add_argument('--loss_w',  default=-10,   type=float, help='factor for non-targeted attack')
  parser.add_argument('--force_bce', action='store_true', help='force use BCE rather than MSE')
  # limit modifiable area
  parser.add_argument('--lim',      default='',    type=str,   help='limit PGD to edge/smap/cam area, or predicted mask by point coord')
  parser.add_argument('--edge_w',   default=0.1,   type=float, help='sobel edge threshold')
  parser.add_argument('--smap_w',   default=0.5,   type=float, help='saliency map threshold')
  parser.add_argument('--cam_w',    default=0.1,   type=float, help='clf attn map threshold')
  parser.add_argument('--cam_meth', default='GradCAM', choices=CAM_METH, help='cam method')
  # misc
  parser.add_argument('--fps',   default=2,      type=int, help='export video FPS, set -1 to disable')
  parser.add_argument('--seed',  default=114514, type=int, help='rand seed')
  parser.add_argument('--debug', action='store_true', help='show detailed pgd log by step')
  parser.add_argument('--vis', action='store_true', help='show generated --lim')
  return parser

def get_args(parser:ArgumentParser) -> Namespace:
  args = get_base_args(parser)
  seed_everything(args.seed)

  # param check
  #assert args.lim in LIM or ',' in args.lim, 'invalid --lim'
  # override
  if args.meth == 'FSGM':
    if args.steps != 1:
      print('warn: force override --steps to 1 for FSGM method')
    args.steps = 1
    args.alpha = args.eps
  # force verbose
  args.debug = True
  return args

def mk_log(args):
  args.log_dp = OUT_PATH / str(datetime.now()).replace(' ', '_').replace(':', '')
  args.log_dp.mkdir()
  save_json(vars(args), args.log_dp / 'args.json')


if __name__ == '__main__':
  parser = get_parser()
  parser.add_argument('--point',     help='point coord formatted as h,w; e.g. 0.3,0.4 or 200,300')
  parser.add_argument('--point_tgt', help='alike --point, but specify target mask point, run targetd attack')
  args = get_args(parser)

  mk_log(args)
  run(args)
