#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/09/06

from atk import *
from atk import _make_smap as make_smap, _make_cam as make_cam

# visualize all --lim for an image

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
  # make tgt
  tgt = make_tgt(ptor, img, args.point_tgt) if args.point_tgt else None
  gc_everything()

  # --lim
  edge = get_edge(img)
  gc_everything()
  smap = make_smap(args, fwd_pack, img, tgt)
  gc_everything()
  cam  = make_cam(args, fwd_pack, img, tgt)
  gc_everything()

  cmap = 'turbo'
  plt.figure(figsize=(10, 6))
  plt.clf()
  plt.subplot(221) ; plt.imshow(img)        ; plt.title('img')
  plt.subplot(222) ; plt.imshow(edge, cmap) ; plt.title(f'edge: {args.edge_w}')
  plt.subplot(223) ; plt.imshow(smap, cmap) ; plt.title(f'smap: {args.smap_w})')
  plt.subplot(224) ; plt.imshow(cam,  cmap) ; plt.title(f'cam: {args.cam_meth} ({args.cam_w})')
  plt.suptitle(f'point: {prompts[0][0]}' + (f'  point_tgt: {args.point_tgt}' if args.point_tgt else ''))
  plt.tight_layout()
  fp = args.log_dp / 'vis_lim.png'
  plt.savefig(fp, dpi=600)
  print(f'>> savefig to {fp}')
  plt.close()

  edge_bin = edge > args.edge_w
  smap_bin = smap > args.edge_w
  cam_bin  = cam  > args.cam_w

  cmap = 'binary'
  plt.figure(figsize=(10, 6))
  plt.clf()
  plt.subplot(221) ; plt.imshow(img)            ; plt.title('img')
  plt.subplot(222) ; plt.imshow(edge_bin, cmap) ; plt.title(f'edge: {args.edge_w}')
  plt.subplot(223) ; plt.imshow(smap_bin, cmap) ; plt.title(f'smap: {args.smap_w}')
  plt.subplot(224) ; plt.imshow(cam_bin,  cmap) ; plt.title(f'{args.cam_meth}: {args.cam_w}')
  plt.suptitle(f'point: {prompts[0][0]}' + (f'  point_tgt: {args.point_tgt}' if args.point_tgt else ''))
  plt.tight_layout()
  fp = args.log_dp / 'vis_lim_binary.png'
  plt.savefig(fp, dpi=600)
  print(f'>> savefig to {fp}')
  plt.close()


if __name__ == '__main__':
  parser = get_parser()
  parser.add_argument('--point',     help='point coord formatted as h,w; e.g. 0.3,0.4 or 200,300')
  parser.add_argument('--point_tgt', help='alike --point, but specify target mask point, run targetd attack')
  args = get_args(parser)

  mk_log(args)
  run(args)
