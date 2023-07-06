#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/06/23 

from utils import *


def run_amg(args, img, sam, mask_gen_kwargs={}):
  amg = SamAutomaticMaskGenerator(sam, **mask_gen_kwargs)
  masks = amg.generate(img)

  show_img(img, masks)

  if not 'results':
    print(len(masks))
    mask0 = masks[0]
    mask0['segmentation']       # [H=534, W=800], bool
    mask0['area']               # int, := mask0['segmentation'].sum()
    mask0['predicted_iou']      # float
    mask0['stability_score']    # float
    mask0['point_coords']       # [[float, float]], 预测出的代表性中心点
    mask0['bbox']               # tuple[4]
    mask0['crop_box']           # tuple[4]

  if 'count map':
    cnt = np.zeros_like(masks[0]['segmentation'], dtype=np.uint8)
    for mask in masks: cnt += mask['segmentation']
    sns.heatmap(cnt)
    plt.suptitle('count map')
    plt.show()


def run(args):
  img = load_img(args.f)  # [H, W, C]
  sam = load_sam(args.M)

  run_amg(args, img, sam)

  if not 'tune params':
    mask_gen_kwargs = {
      'points_per_side': 32,
      'pred_iou_thresh': 0.86,
      'stability_score_thresh': 0.92,
      'crop_n_layers': 1,
      'crop_n_points_downscale_factor': 2,
      'min_mask_region_area': 100,            # Requires open-cv to run post-processing
    }
    run_amg(args, img, sam, mask_gen_kwargs)


if __name__ == '__main__':
  run(get_args())
