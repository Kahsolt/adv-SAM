#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/08/15

from atk import *
from atk import _make_cam as make_cam

# visualize comparing cam on clean & adv image

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
  tgt = make_tgt(ptor, img, args.point_tgt) if args.point_tgt else None
  # make lim
  lim = make_lim(args, img, tgt, ptor_pack, fwd_pack) if args.lim else None

  # cam X
  cam = make_cam(args, fwd_pack, img, tgt)
  gc_everything()

  # pred X
  mask, piou = make_pred(ptor_pack, img)

  # attack
  adv, mask_adv, piou_adv = pgd(args, fwd_pack, img, tgt=tgt, lim=lim)
  # cam AX
  cam_adv = make_cam(args, fwd_pack, adv, tgt)
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
  parser.add_argument('--point',     help='point coord formatted as h,w; e.g. 0.3,0.4 or 200,300')
  parser.add_argument('--point_tgt', help='alike --point, but specify target mask point, run targetd attack')
  args = get_args(parser)

  run(args)
