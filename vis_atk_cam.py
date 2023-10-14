#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/08/15

# visualize comparing cam on clean & adv image

from atk import *

CAM_METH = ['GradCAM', 'HiResCAM', 'ScoreCAM', 'GradCAMPlusPlus', 'AblationCAM', 'XGradCAM', 'EigenCAM', 'FullGrad']


@torch.no_grad()
def make_cam(args, fwd_pack:FwdPack, img:npimg_u8, tgt:npimg_b1, use_cpu:bool=True) -> npimg_f32:
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
  Y, _ = make_Y(args, tgt, img, X.device)   # [B=1, H, W]
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
  ptor_pack = ptor, prompts
  fwd_pack = fwder, prompts, loss_fn
  # make tgt
  tgt = make_tgt(ptor, img, args.point_tgt)

  # cam X
  cam = make_cam(args, fwd_pack, img, tgt, use_cpu=True)

  # pred X
  mask, piou = make_pred(ptor_pack, img)

  # attack
  adv, mask_adv, piou_adv, _ = pgd(args, fwd_pack, img, tgt=tgt)
  # cam AX
  cam_adv = make_cam(args, fwd_pack, adv, tgt, use_cpu=True)
  # pred AX
  if not 'loopback to predictor':
    mask_adv, piou_adv = make_pred(ptor_pack, adv)

  if 'show':
    cmap_hot = 'turbo'
    cmap_bin = 'binary'
    plt.figure(figsize=(10, 6))
    plt.clf()
    plt.subplot(231) ; plt.imshow(img)                ; plt.title('img')
    plt.subplot(232) ; plt.imshow(cam,      cmap_hot) ; plt.title(f'cam: {args.cam_meth}')
    plt.subplot(233) ; plt.imshow(mask,     cmap_bin) ; plt.title(f'mask (piou={piou:.5f})')
    plt.subplot(234) ; plt.imshow(adv)                ; plt.title('adv')
    plt.subplot(235) ; plt.imshow(cam_adv,  cmap_hot) ; plt.title(f'cam_adv: {args.cam_meth}')
    plt.subplot(236) ; plt.imshow(mask_adv, cmap_bin) ; plt.title(f'mask_adv (piou={piou_adv:.5f})')
    plt.suptitle(f'point: {prompts[0][0]}' + (f'  point_tgt: {args.point_tgt}' if args.point_tgt else ''))
    plt.tight_layout()
    fp = args.log_dp / 'atk_vis.png'
    plt.savefig(fp, dpi=600)
    print(f'>> savefig to {fp}')
    plt.close()


if __name__ == '__main__':
  parser = get_parser()
  parser.add_argument('--cam_meth', default='GradCAM', choices=CAM_METH, help='cam method')
  parser.add_argument('--point',     help='point coord formatted as h,w; e.g. 0.3,0.4 or 200,300')
  parser.add_argument('--point_tgt', help='alike --point, but specify target mask point, run targetd attack')
  args = get_args(parser)

  mk_log(args)
  run(args)
