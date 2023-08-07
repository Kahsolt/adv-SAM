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
  from pycocotools.mask import decode
  HAS_PYCOCOTOOLS = True
except ImportError:
  print('>> [warn] missing lib "pycocotools", will not run attack over dataset')
  HAS_PYCOCOTOOLS = False
try:
  from moviepy.editor import ImageSequenceClip
  HAS_MOVIEPY = True
except ImportError:
  print('>> [warn] missing lib "moviepy", will not generate adv pred step by step')
  HAS_MOVIEPY = False

from utils import *
from utils import get_parser as get_base_parser
from segment_anything.modeling import Sam
from segment_anything.utils.transforms import ResizeLongestSide

OUT_PATH = Path('out') ; OUT_PATH.mkdir(exist_ok=True)
INTERP_MODE = 'bilinear'      # 'nearest'

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
def pgd(args, fwder:SamForwarder, prompts:tuple, img:npimg_u8, tgt:npimg_b1=None, lim:npimg_b1=None, log:bool=True) -> Tuple[npimg_u8, npimg_b1, float]:
  is_tgt = tgt is not None
  is_lim = lim is not None
  is_gen_vid = HAS_MOVIEPY and args.fps > 0

  is_01      = lambda x: -1e-5 <= x.min() and x.max() <= 1.0
  b1_to_u8   = lambda x: np.tile((np.expand_dims(x, -1) * 255).astype(np.uint8), reps=(1, 1, 3))
  norm       = lambda x: fwder.norm_image  (x  * 255)    # [0, 1] to normed
  denorm     = lambda x: fwder.denorm_image(x) / 255.0   # normed to [0, 1]
  _cvt       = lambda x: torch.as_tensor(x).round().clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
  decode_img = lambda x: _cvt(fwder.unresize_image(fwder.denorm_image(x))[0])
  decode_dx  = lambda x: _cvt(minmax_norm(fwder.unresize_image(x.abs())[0], vmax=args.eps) * 255)
  decode_msk = lambda x: _cvt(x * 255)

  if 'preprocess raw input':
    # NOTE: must call `transform_image` first to set up `self.original_size`
    X = fwder.transform_image(img)      # [B, C=3, pH, pW], var-mean normed image, ~ [-2.4, 2.4]
    Xo = (denorm(X) * 255).byte().div(255)  # [B, C=3, pH, pW], ~ [0.0, 1.0]
    if is_lim:
      M = fwder.transform_image(b1_to_u8(lim), is_edge=True).bool()   # [B, C=3, pH, pW]
    else:
      M = 1.0
    P = fwder.transform_prompts(*prompts)

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
  
  if is_gen_vid:
    dxs, preds = [], []

  for _ in (tqdm if log else enumerate)(range(args.steps)):
    AX.requires_grad = True

    logits, piou = fwder.forward(AX, *P)    # [B=1, H, W], [B=1]
    mask = logits > fwder.model.mask_threshold
    if is_tgt:
      loss = F.binary_cross_entropy_with_logits(logits, Y)
    else:
      loss = F.mse_loss(logits, Y)

    g = grad(loss, AX, loss)[0]
    delta = g.sign() * M * args.alpha

    AX = denorm(AX.detach()) - delta
    DX: Tensor = (AX - Xo).clamp(-args.eps, args.eps)
    AX = norm((Xo + DX).detach().clamp(0.0, 1.0))

    if is_gen_vid:
      dxs.append(decode_dx(DX))
      preds.append(np.tile(decode_msk(mask), reps=(1, 1, 3)))

    if log:
      print(f'>> piou: {piou.item():.5f}, masked_area: {(logits > 0).sum() / logits.numel():.3%}')

  if is_gen_vid:
    try:
      dxs_rep   = dxs   + [dxs  [-1]] * args.fps * 2
      preds_rep = preds + [preds[-1]] * args.fps * 2
      ImageSequenceClip(dxs_rep,   fps=args.fps).write_videofile(str(args.log_dp / 'pgd_noise.mp4'))
      ImageSequenceClip(preds_rep, fps=args.fps).write_videofile(str(args.log_dp / 'pgd_mask.mp4'))
    except:
      print_exc()
  
  if log:
    d: Tensor = torch.abs(Xo - denorm(AX))
    print('Linf (raw):', d.max().item())
    print('L1 (raw):', d.mean().item())

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

def make_prompts(point:Union[str, ndarray], img_size:tuple) -> Tuple:
  if isinstance(point, str):
    point = _parse_point(point, img_size)
    print(f'>> point: {point}')
    coords = np.expand_dims(np.asarray(point, dtype=np.int32), axis=0)
  else:
    coords = point
  labels = np.asarray([1])
  return (coords, labels, None, None)

def make_pred(ptor:SamPredictor, img:npimg_u8, prompts:Tuple, multi_out:bool=False) -> Union[Tuple[npimg_b1, float], Tuple[List[npimg_b1], List[float]]]:
  ptor.reset_image()
  ptor.set_image(img)
  mask, piou, _ = ptor.predict(*prompts, multimask_output=multi_out)
  if multi_out:
    return [mask[i] for i in range(len(mask))], piou.tolist()
  else:
    return mask[0], piou.item()    # [C=1, H, W] => [H, W]

def make_diff(img:npimg_u8, adv:npimg_u8) -> npimg_f32:
  im0 = img / 255.0
  im1 = adv / 255.0
  d: npimg_f32 = np.abs(im0 - im1)
  print('Linf (proc):', d.max())
  print('L1 (proc):', d.mean())
  diff = minmax_norm(d)
  return diff


def run_single(args):
  # meta log
  args.log_dp = OUT_PATH / str(datetime.now()).replace(' ', '_').replace(':', '')
  args.log_dp.mkdir()
  fp = args.log_dp / 'args.json'
  save_json(vars(args), fp)

  # model
  sam = load_sam(args.M)
  ptor = SamPredictor(sam)
  fwder = SamForwarder(sam)
  # image
  img = load_img(args.f)  # [H, W, C]
  img_size = img.shape[:-1]

  # make lim
  if args.lim:
    if args.lim == 'edge':
      lim = get_mask_edge(img, args.thresh)
    else:   # it should be a point coord
      prompts = make_prompts(args.lim, img_size)
      mask, _ = make_pred(ptor, img, prompts)
      lim = mask > sam.mask_threshold
  else:
    lim = None
  
  # make target
  if args.point_tgt:
    prompts_tgt = make_prompts(args.point_tgt, img_size)
    tgt, _ = make_pred(ptor, img, prompts_tgt)
  else:
    tgt = None

  # make prompts
  prompts = make_prompts(args.point, img_size)

  # pred X
  mask, piou = make_pred(ptor, img, prompts)

  # attack
  adv, mask_adv, piou_adv = pgd(args, fwder, prompts, img, tgt=tgt, lim=lim)
  diff = make_diff(img, adv)

  # pred AX
  if not 'loopback to predictor':
    mask_adv, piou_adv = make_pred(ptor, adv, prompts)

  if 'show':
    plt.figure(figsize=(10, 6))
    plt.clf()
    plt.subplot(231) ; plt.imshow(img)                  ; plt.title('img')
    plt.subplot(232) ; plt.imshow(mask, cmap='gray')    ; plt.title(f'mask (piou={piou:.5f})')
    if lim is not None:
      plt.subplot(233) ; plt.imshow(lim, cmap='gray')   ; plt.title('lim')
    plt.subplot(234) ; plt.imshow(adv)                   ; plt.title('img_adv')
    plt.subplot(235) ; plt.imshow(mask_adv, cmap='gray') ; plt.title(f'mask_adv (piou={piou_adv:.5f})')
    plt.subplot(236) ; plt.imshow(diff, cmap='gray')     ; plt.title('diff (proc)')
    plt.suptitle(f'point: {prompts[0][0]}')
    plt.tight_layout()
    fp = args.log_dp / 'atk_sam.png'
    plt.savefig(fp, dpi=600)
    print(f'>> savefig to {fp}')
    plt.close()


@timer
def run_dataset(args):
  # meta
  assert args.lim in ['', 'edge', 'tgt']
  if args.debug:
    args.log_dp = OUT_PATH / 'tmp_dataset'
    args.log_dp.mkdir(exist_ok=True)
  else:
    args.fps = -1
  seed_everything(args.seed)

  sample_ids = sorted({fp.stem for fp in DATA_ROOT.iterdir()})
  np.random.shuffle(sample_ids)
  if args.limit_img > 0:
    sample_ids = sample_ids[:args.limit_img]
  
  sam = load_sam(args.M)
  ptor = SamPredictor(sam)
  fwder = SamForwarder(sam)

  hist: List = load_json(HIST_FILE, list)
  s = time()

  iou_sum, iou_cnt = 0.0, 0
  interrupted = False
  try:
    for id in tqdm(sample_ids):
      img = load_img(DATA_ROOT / f'{id}.jpg')
      img_size = img.shape[:-1]

      cfg = load_cfg(DATA_ROOT / f'{id}.json')
      annots = cfg['annotations']
      if args.limit_ant > 0:
        annots_sel = np.random.choice(annots, size=args.limit_ant, replace=False)
      else:
        annots_sel = annots
      for annot in annots_sel:
        if 'input':
          point = np.asarray(annot['point_coords'])
          prompts = make_prompts(point, img_size)
        if 'ground truth':
          mask_gt: npimg_b1 = np.ascontiguousarray(decode(annot['segmentation']), dtype=bool)
          iou_gt: float = annot['predicted_iou']
        
        if args.atk:
          if args.tgt:
            annots_tgt = np.random.choice(annots, size=1, replace=False)[0]
            point_tgt = np.asarray(annots_tgt['point_coords'])
            prompts_tgt = make_prompts(point_tgt, img_size)
            tgt, _ = make_pred(ptor, img, prompts_tgt)
          else:
            tgt = None
        
          if args.lim == 'edge':
            lim = get_mask_edge(img, args.thresh)
          if args.lim == 'tgt':
            lim = tgt
          else:
            lim = None

          _, mask_hat, piou = pgd(args, fwder, prompts, img, tgt, lim, log=args.debug)
        else:
          mask_hat, piou = make_pred(ptor, img, prompts, multi_out=args.multi_mask)

        if isinstance(mask_hat, list):
          riou = max([get_iou(m, mask_gt) for m in mask_hat])
        else:
          riou = get_iou(mask_hat, mask_gt)

        iou_cnt += 1
        iou_sum += riou

  except KeyboardInterrupt:
    print('>> interrupted!!')
    interrupted = True
  except:
    print_exc()
  finally:
    miou = 0.0 if iou_cnt == 0 else (iou_sum / iou_cnt)
    print(f'>> miou: {miou}')

    t = time()
    rec = {
      'miou': miou,
      'interrupted': interrupted,
      'ts': t - s,
      'ts_start': str(datetime.fromtimestamp(t)),
      'ts_finish': str(datetime.fromtimestamp(s)),
      'args': vars(args),
    }
    hist.insert(0, rec)
    save_json(hist, HIST_FILE)


if __name__ == '__main__':
  parser = get_base_parser()
  # single image
  parser.add_argument('--point',     help='point coord formatted as h,w; e.g. 0.3,0.4 or 200,300')
  parser.add_argument('--point_tgt', help='alike --point, but specify target mask point, run targetd attack')
  parser.add_argument('--lim',    default='',    type=str,   help='limit PGD to edge area, or a predicted mask area')
  parser.add_argument('--thresh', default=0.1,   type=float, help='edge threshold')
  parser.add_argument('--steps',  default=40,    type=int)
  parser.add_argument('--eps',    default=8/255, type=float)
  parser.add_argument('--alpha',  default=1/255, type=float)
  parser.add_argument('--loss_w', default=-10,   type=float, help='factor for non-targeted attack')
  parser.add_argument('--fps',    default=2,     type=int, help='export video FPS, set -1 to disable')
  # whole dataset
  parser.add_argument('-L', '--limit_img', default=-1, type=int, help='limit run image count, set -1 for all')
  parser.add_argument('-K', '--limit_ant', default=1,  type=int, help='limit run annot count of each image, set -1 for all')
  parser.add_argument('--atk', action='store_true', help='enable PGD attack')
  parser.add_argument('--tgt', action='store_true', help='enable targeted attack (use randomly another mask as the target)')
  parser.add_argument('--multi_mask', action='store_true', help='use essay method to calc mIoU (pick the highest IoU from multipile mask outputs)')
  parser.add_argument('--seed', default=114514, type=int, help='rand seed')
  parser.add_argument('--debug', action='store_true', help='show detailed log step by step')
  args = get_args(parser)

  if args.f == DATASET_LITERAL:
    assert HAS_PYCOCOTOOLS, 'run "pip install pycocotools" first!!'
    run_dataset(args)
  else:
    args.debug = True
    run_single(args)
