#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/07/05 

import random
from tqdm import tqdm
from enum import Enum
from datetime import datetime
from pprint import pprint as pp
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
from segment_anything.modeling import Sam
from segment_anything.utils.transforms import ResizeLongestSide

class AtkMeth(Enum):
  FGSM   = 'FGSM'
  PGD    = 'PGD'
  SegPGD = 'SegPGD'

class AtkLoss(Enum):
  MAE     = 'MAE'
  MSE     = 'MSE'
  ClipMAE = 'ClipMAE'
  ClipMSE = 'ClipMSE'
  BCE     = 'BCE'

ATK_METH = [x.value for x in AtkMeth]
ATK_LOSS = [x.value for x in AtkLoss]
CAM_METH = ['GradCAM', 'HiResCAM', 'ScoreCAM', 'GradCAMPlusPlus', 'AblationCAM', 'XGradCAM', 'EigenCAM', 'FullGrad']
LIM      = ['', 'edge', 'smap', 'cam', 'tgt']   # or a point coord, use prefix '~' to negate
SAM_MASK_THRESH = 0.0

class SamForwarder(nn.Module):

  def __init__(self, sam:Sam):
    super().__init__()

    assert SAM_MASK_THRESH == sam.mask_threshold, f'sam.mask_threshold ({sam.mask_threshold}) != SAM_MASK_THRESH ({SAM_MASK_THRESH})'

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

def make_loss_fn(args):
  loss_fns = {
    AtkLoss.MAE:     F.l1_loss,
    AtkLoss.MSE:     F.mse_loss,
    AtkLoss.ClipMAE: lambda o, y: F.l1_loss (torch.clamp_min(o, args.loss_w), y),
    AtkLoss.ClipMSE: lambda o, y: F.mse_loss(torch.clamp_min(o, args.loss_w), y),
    AtkLoss.BCE:     F.binary_cross_entropy_with_logits,
  }
  if args.loss not in loss_fns: raise ValueError(f'unknown loss fn: {args.loss.value}')
  return loss_fns[args.loss]

FwdPack = Tuple[SamForwarder, Prompts, Callable]    # loss_fn
PtorPack = Tuple[SamPredictor, Prompts]    # loss_fn

# ↓↓↓ repo\pytorch-grad-cam\pytorch_grad_cam\base_cam.py ↓↓↓

from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def BaseCAM_forward_hijack(self: BaseCAM, input_tensor: torch.Tensor, targets: List[torch.nn.Module], eigen_smooth: bool = False) -> np.ndarray:
  if self.cuda: input_tensor = input_tensor.cuda()

  if self.compute_input_gradient:     # False
    input_tensor = torch.autograd.Variable(input_tensor, requires_grad=True)

  outputs = self.activations_and_grads(input_tensor)      # <= VRAM consuming
  if targets is None:
    target_categories = np.argmax(outputs.cpu().data.numpy(), axis=-1)
    targets = [ClassifierOutputTarget(category) for category in target_categories]
  
  if self.uses_gradients:     # True
    self.model.zero_grad()
    loss = sum([target(output) for target, output in zip(targets, outputs)])
    loss.backward(retain_graph=True)

  cam_per_layer = self.compute_cam_per_layer(input_tensor, targets, eigen_smooth)
  cam_agg = self.aggregate_multi_layers(cam_per_layer)

  # NOTE: cannot find the vram leak?!!
  self.model.zero_grad()
  self.target_layers.clear()
  self.activations_and_grads.release()
  self.activations_and_grads.handles.clear()
  self.activations_and_grads.activations.clear()
  self.activations_and_grads.gradients.clear()
  gc_everything()

  return cam_agg

# ↑↑↑ repo\pytorch-grad-cam\pytorch_grad_cam\base_cam.py ↑↑↑


@torch.no_grad()
def pgd(args, fwd_pack:FwdPack, img:npimg_u8, tgt:npimg_b1=None, lim:npimg_b1=None, 
        multi_mask:bool=False, log:bool=True) -> Union[Tuple[npimg_u8, npimg_b1, float], Tuple[npimg_u8, List[npimg_b1], List[float]]]:

  fwder, prompts, loss_fn = fwd_pack
  is_tgt = tgt is not None
  is_lim = lim is not None

  b1_to_u8   = lambda x: np.tile((np.expand_dims(x, -1) * 255).astype(np.uint8), reps=(1, 1, 3))
  norm       = lambda x: fwder.norm_image  (x  * 255)    # [0, 1] to normed
  denorm     = lambda x: fwder.denorm_image(x) / 255.0   # normed to [0, 1]
  _cvt       = lambda x: torch.as_tensor(x).round().clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
  decode_img = lambda x: _cvt(fwder.unresize_image(fwder.denorm_image(x))[0])
  decode_dx  = lambda x: _cvt(minmax_norm(fwder.unresize_image(x.abs())[0], vmax=args.eps) * 255)
  decode_msk = lambda x: _cvt(x * 255)

  # NOTE: must call `transform_image(img)` first to set up `self.original_size`, before other `transform_*`
  X = fwder.transform_image(img)          # [B, C=3, pH, pW], var-mean normed image, vrng ~[-2.4, 2.4]
  Xo = (denorm(X) * 255).byte().div(255)  # [B, C=3, pH, pW], vrng ~[0.0, 1.0]
  M = fwder.transform_image(b1_to_u8(lim), is_edge=True).bool() if is_lim else 1.0  # [B, C=3, pH, pW]
  P = fwder.transform_prompts(*prompts)

  # make target tensor
  if is_tgt:   # tagerted, load target mask (npimg_b1)
    Y_bin = torch.from_numpy(tgt).to(X.device)
    Y_bin.unsqueeze_(0)   # [B, H, W]
    if args.loss == AtkLoss.BCE:
      Y = Y_bin.float()
    else:
      Y = (Y_bin * args.loss_v + ~Y_bin * args.loss_w).float()
  else:                 # non-tagerted, generate target logits
    H, W, _ = img.shape
    if args.loss == AtkLoss.BCE:  # non-tagert for BCE is all zeros
      Y = torch.zeros([H, W]).to(X.device, torch.float32)
    else:                         # otherwise is same-valued logits (as background)
      Y = torch.ones([H, W]).to(X.device, torch.float32) * args.loss_w
    Y.unsqueeze_(0)   # [B, H, W]
    Y_bin = Y.bool()
  assert Y_bin.dtype in ['bool', bool, torch.bool]

  # random start
  if args.meth != AtkMeth.FGSM:
    noise = torch.empty_like(X).uniform_(-args.eps, args.eps) * M
    AX = norm((Xo + noise).clamp(0.0, 1.0))
  else:
    AX = X.clone()

  is_gen_vid = HAS_MOVIEPY and args.fps > 0 and not args.nolog
  if is_gen_vid: dxs, preds = [], []

  for i in (tqdm if log else list)(range(args.steps)):
    AX.requires_grad = True

    with torch.enable_grad():
      logits, piou = fwder.forward(AX, *P)    # [B=1, H, W], [B=1]
      mask = logits > fwder.model.mask_threshold

      if args.meth == AtkMeth.SegPGD:
        lmbd = i / (args.steps * 2)
        attacked = mask[0] == Y_bin
        loss_t = loss_fn( attacked * logits, Y) *      lmbd
        loss_f = loss_fn(~attacked * logits, Y) * (1 - lmbd)
        loss = loss_t + loss_f
      elif args.meth in [AtkMeth.FGSM, AtkMeth.PGD]:
        loss = loss_fn(logits, Y)
      else:
        raise ValueError(f'unknown attack meth: {args.meth.value}')

    #if loss.abs() < 1e-5: break     # NOTE: stop early :)
    g = grad(loss, AX, loss)[0]
    delta = g.sign() * M * args.alpha

    AX = denorm(AX) - delta
    DX: Tensor = (AX - Xo).clamp(-args.eps, args.eps)
    AX = norm((Xo + DX).clamp(0.0, 1.0))

    if is_gen_vid:
      dxs  .append(decode_dx(DX))
      preds.append(np.tile(decode_msk(mask), reps=(1, 1, 3)))

    if log: print(f'>> loss: {loss.item():.5f}, piou: {piou.item():.5f}, masked_area: {(logits > 0).sum() / logits.numel():.3%}')

  fwder.model.zero_grad()
  gc_everything()

  if is_gen_vid:
    try:
      dxs_rep   = [dxs  [0]] * args.fps + dxs   + [dxs  [-1]] * args.fps * 4
      preds_rep = [preds[0]] * args.fps + preds + [preds[-1]] * args.fps * 4
      ImageSequenceClip(dxs_rep,   fps=args.fps).write_videofile(str(args.log_dp / 'pgd_noise.mp4'))
      ImageSequenceClip(preds_rep, fps=args.fps).write_videofile(str(args.log_dp / 'pgd_mask.mp4'))
    except:
      print_exc()
  
  if log:
    d: Tensor = torch.abs(Xo - denorm(AX))
    print('Linf (raw):', d.max() .item())
    print('L1 (raw):',   d.mean().item())
  
  logits, piou = fwder.forward(AX, *P, multi_mask=multi_mask)
  mask = logits > fwder.model.mask_threshold
  if multi_mask: return decode_img(AX), [mask[i].cpu().numpy() for i in range(len(mask))], piou.tolist()
  else:          return decode_img(AX), mask[0].cpu().numpy(), piou.item()


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
  if isinstance(point, str) or point is None:
    point = _parse_point(point, img_size)
    print(f'>> point: {point}')
    coords = np.expand_dims(np.asarray(point, dtype=np.int32), axis=0)
  else:
    coords = point
  labels = np.asarray([1], dtype=np.int32)
  return (coords, labels, None, None)

def make_pred(ptor_pack:PtorPack, img:npimg_u8, multi_mask:bool=False, ret_logits:bool=False) -> Union[Tuple[npimg_b1, float], Tuple[List[npimg_b1], List[float]]]:
  ptor, prompts = ptor_pack
  ptor.set_image(img)
  mask, piou, _ = ptor.predict(*prompts, multimask_output=multi_mask, return_logits=ret_logits)
  ptor.reset_image()
  if multi_mask:
    return [mask[i] for i in range(len(mask))], piou.tolist()
  else:
    return mask[0], piou.item()    # [C=1, H, W] => [H, W]

def make_tgt(ptor:SamPredictor, img:npimg_u8, point_tgt:Union[str, ndarray])-> npimg_b1:
  if point_tgt is None or point_tgt == '': return None

  img_size = img.shape[:-1]
  prompts_tgt = make_prompts(point_tgt, img_size)
  tgt, _ = make_pred((ptor, prompts_tgt), img)
  return tgt

def make_Y(tgt:npimg_b1=None, img:npimg_u8=None, w:float=-10.0) -> Tensor:
  '''
    non-targeted: (img.shape, loss_w) -> Tensor
    targeted: npimg_b1 -> Tensor
  '''

  if tgt is not None:   # targeted
    # value in {0, 1}
    Y = torch.from_numpy(tgt).float()
  else:                 # non-targeted
    # value in {w}, where w is a large negative
    H, W, _ = img.shape
    Y = torch.ones([H, W]) * w
  return Y.unsqueeze_(0)    # [C=1, oH, oW]

@torch.no_grad()
def _make_smap(args, fwd_pack:FwdPack, img:npimg_u8, tgt:npimg_b1) -> npimg_f32:
  decode = lambda x: fwder.unresize_image(x)[0].permute(1, 2, 0).cpu().numpy()

  fwder, prompts, loss_fn = fwd_pack
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
  smap: npimg_f32 = decode(psmap)   # [H, W, C=3]
  smap = smap.mean(axis=-1)         # [H, W]
  return smap

@torch.no_grad()
def _make_cam(args, fwd_pack:FwdPack, img:npimg_u8, tgt:npimg_b1, use_cpu:bool=False) -> npimg_f32:
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
  
  fwder, prompts, loss_fn = fwd_pack
  decode = lambda x: fwder.unresize_image(x)[0].permute(1, 2, 0).cpu().numpy()

  X = fwder.transform_image(img)
  if use_cpu: X = X.cpu()
  P = fwder.transform_prompts(*prompts)
  if use_cpu: P = [p.cpu() if isinstance(p, Tensor) else p for p in P]
  Y = make_Y(tgt, img, args.loss_w).to(X.device)   # [B=1, H, W]
  if use_cpu: fwder = fwder.cpu()
  L = [fwder.model.mask_decoder.output_upscaling]
  T = [SAMTarget(Y[0], loss_fn)]

  # hijack .forward() bind args
  fwder.forward_orignal = fwder.forward
  fwder.forward = lambda x: fwder.forward_orignal(x, *P)[0]

  cam_inst: BaseCAM = locals()[args.cam_meth](fwder, L)
  #cam_inst.forward = lambda *args, **kwargs: BaseCAM_forward_hijack(cam_inst, *args, **kwargs)
  with torch.enable_grad():
    cam = cam_inst(input_tensor=X, targets=T, eigen_smooth=False)

  # unhijack .forward()
  fwder.forward = fwder.forward_orignal

  del X, P, Y, L, T
  a_g: ActivationsAndGradients = cam_inst.activations_and_grads
  a_g.activations.clear()
  a_g.gradients  .clear()
  a_g.handles    .clear()
  del cam_inst, a_g

  if use_cpu: fwder = fwder.to(device)

  cam_t = torch.from_numpy(cam).unsqueeze_(1)
  cam: npimg_f32 = decode(cam_t)  # [H, W, C=3]
  cam = cam.mean(axis=-1)         # [H, W]
  return cam

def make_lim(args, img:npimg_u8, tgt:npimg_b1, ptor_pack:PtorPack=None, fwd_pack:FwdPack=None) -> npimg_b1:
  if not args.lim: return None

  lim_s: str = args.lim
  inv = False
  if lim_s.startswith('~'):
    lim_s = lim_s[1:]
    inv = True

  if   lim_s == 'edge': lim =  get_edge (                img     ) > args.edge_w
  elif lim_s == 'smap': lim = _make_smap(args, fwd_pack, img, tgt) > args.smap_w
  elif lim_s == 'cam':  lim = _make_cam (args, fwd_pack, img, tgt) > args.cam_w
  elif lim_s == 'tgt':  lim = tgt
  else:   # it should be a point coord
    ptor, _ = ptor_pack
    prompts = make_prompts(lim_s, img.shape[:-1])
    mask, _ = make_pred((ptor, prompts), img)
    lim = mask > ptor.model.mask_threshold

  if inv: lim = ~lim

  if args.show:
    #plt.clf()
    #visual = show_cam_on_image(img, cam, use_rgb=True)
    #plt.imshow(visual)
    #plt.show()
    
    plt.clf()
    plt.subplot(121) ; plt.title('img') ; plt.imshow(img)
    plt.subplot(122) ; plt.title('lim') ; plt.imshow(lim)
    plt.show()

  return lim


def run(args):
  # model & loss
  sam = load_sam(args.M)
  ptor = SamPredictor(sam)
  fwder = SamForwarder(sam)
  loss_fn = make_loss_fn(args)

  # image
  img = load_img(args.f)  # [H, W, C]
  # make prompts
  prompts = make_prompts(args.point, img.shape[:-1])
  fwd_pack = fwder, prompts, loss_fn
  ptor_pack = ptor, prompts
  # make tgt
  img_tgt = load_img(args.f_tgt) if args.f_tgt else img
  assert img.shape == img_tgt.shape, f'>> image shape of src and tgt mismatch: {img.shape} != {img_tgt.shape}, current not supported :('
  tgt = make_tgt(ptor, img_tgt, args.point_tgt)
  is_tgt = tgt is not None
  # make lim
  lim = make_lim(args, img, tgt, ptor_pack, fwd_pack)
  is_lim = lim is not None

  # pred X
  mask, piou = make_pred(ptor_pack, img)
  # attack
  adv, mask_adv, piou_adv = pgd(args, fwd_pack, img, tgt=tgt, lim=lim)
  # delta
  diff = make_diff(img, adv)
  # pred AX
  if not 'loopback to predictor':
    mask_adv, piou_adv = make_pred(ptor_pack, adv)

  if 'show':
    cmap = 'gray'
    plt.figure(figsize=(10, 6))
    plt.clf()
    plt.subplot(231) ; plt.imshow(img)            ; plt.title('img')
    plt.subplot(232) ; plt.imshow(mask, cmap)     ; plt.title(f'mask (piou={piou:.5f})')
    plt.subplot(233)
    if   is_lim:       plt.imshow(lim, cmap)      ; plt.title(f'lim: {args.lim}')
    elif is_tgt:       plt.imshow(tgt, cmap)      ; plt.title('tgt')
    plt.subplot(234) ; plt.imshow(adv)            ; plt.title('img_adv')
    plt.subplot(235) ; plt.imshow(mask_adv, cmap) ; plt.title(f'mask_adv (piou={piou_adv:.5f})')
    plt.subplot(236) ; plt.imshow(diff, cmap)     ; plt.title('diff (proc)')
    plt.suptitle(f'point: {prompts[0][0]}')
    plt.tight_layout()
    if args.nolog:
      plt.show()
    else:
      fp = args.log_dp / 'atk_sam.png'
      plt.savefig(fp, dpi=600)
      print(f'>> savefig to {fp}')
    plt.close()


def get_parser() -> ArgumentParser:
  from utils import get_parser as get_base_parser

  parser = get_base_parser()
  # pgd params
  parser.add_argument('--meth',   default='PGD', choices=ATK_METH, help='attack method')
  parser.add_argument('--loss',   default='MSE', choices=ATK_LOSS, help='attack creterion')
  parser.add_argument('--loss_w', default=-10,   type=float, help='target outval for non-targeted and targeted-bg, for any loss_fn except BCE')
  parser.add_argument('--loss_v', default=10,    type=float, help='target outval for targeted-fg, for any loss_fn except BCE')
  parser.add_argument('--steps',  default=40,    type=int)
  parser.add_argument('--eps',    default=8/255, type=float)
  parser.add_argument('--alpha',  default=1/255, type=float)
  # limit modifiable area
  parser.add_argument('--lim',      default='',    type=str,   help=f'limit PGD to edge/smap/cam area, choose value from {LIM}; or predicted mask area, set to a point coord')
  parser.add_argument('--edge_w',   default=0.1,   type=float, help='sobel edge threshold')
  parser.add_argument('--smap_w',   default=0.5,   type=float, help='saliency map threshold')
  parser.add_argument('--cam_w',    default=0.1,   type=float, help='clf attn map threshold')
  parser.add_argument('--cam_meth', default='GradCAM', choices=CAM_METH, help='cam method')
  # misc
  parser.add_argument('--fps',   default=2,      type=int, help='export video FPS, set -1 to disable')
  parser.add_argument('--seed',  default=114514, type=int, help='rand seed')
  parser.add_argument('--nolog', action='store_true', help='do not save logs')
  parser.add_argument('--debug', action='store_true', help='show detailed pgd log by step')
  parser.add_argument('--show',  action='store_true', help='show generated --lim')
  return parser

def get_args(parser:ArgumentParser) -> Namespace:
  from utils import get_args as get_base_args

  # fix seed
  args = get_base_args(parser)
  seed_everything(args.seed)

  # type convert
  args.meth = AtkMeth(args.meth)
  args.loss = AtkLoss(args.loss)

  # force override
  if args.meth == AtkMeth.FGSM:
    if args.steps != 1:
      print('warn: force override --steps for FGSM method')
      args.steps = 1
    if args.alpha != args.eps:
      print('warn: force override --alpha for FGSM method')
      args.alpha = args.eps

  # force verbose for single run
  args.debug = True

  return args

def mk_log(args):
  if args.nolog:
    args.fps = -1
  else:
    args.argv = ' '.join(sys.argv)
    args.log_dp = OUT_PATH / str(datetime.now()).replace(' ', '_').replace(':', '')
    args.log_dp.mkdir()
    save_json(vars(args), args.log_dp / 'args.json')
  pp(vars(args))


if __name__ == '__main__':
  parser = get_parser()
  parser.add_argument('--point',     help='point coord formatted as h,w; e.g. 0.3,0.4 or 200,300')
  parser.add_argument('--point_tgt', help='alike --point, but specify target mask point to run targeted attack; default image is -f')
  parser.add_argument('--f_tgt',     help='alike -f, but specify target image filepath, change image of --point_tgt')
  args = get_args(parser)

  if args.f_tgt: assert Path(args.f_tgt).is_file(), '--f_tgt is not a valid filepath'
  #if args.point_tgt: assert args.loss == AtkLoss.BCE, 'targeted attack only supports --loss BCE'

  mk_log(args)
  run(args)
