#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/06/23 

from utils import *
import seaborn as sns


def run_sam(args, sam, img):
  X = torch.from_numpy(img).permute([2, 0, 1])  # [0, 255]
  print('X.shape:', X.shape)

  inputs = [{
    'image': X.to(device),
    'original_size': X.shape[1:],
  }]
  ret = sam(inputs)[0]
  ret['masks']                # [B=1, C=3, H=534, W=800], bool
  ret['low_res_logits']       # [B=1, C=3, H=256, W=256], float
  ret['iou_predictions']      # [B=1, D=3], float
  breakpoint()


def run_ptor(args, sam, img):
  predictor = SamPredictor(sam)
  predictor.set_image(img)

  if 'point a coord':
    coords = np.asarray([[200, 400]])
    labels = np.asarray([1])
  masks, iou_predictions, low_res_masks = predictor.predict(point_coords=coords, point_labels=labels)

  for mask, piou, lmask in zip(masks, iou_predictions, low_res_masks):
    print('piou:', piou.item())

    plt.clf()
    plt.subplot(121) ; sns.heatmap( mask) ; plt.title('mask')
    plt.subplot(122) ; sns.heatmap(lmask) ; plt.title('mask orig.')
    plt.suptitle(f'piou: {piou.item()}')
    plt.show()


def run_amg(args, sam, img, override_params=False):
  if override_params:
    kwargs = {
      'points_per_side': 32,
      'pred_iou_thresh': 0.86,
      'stability_score_thresh': 0.92,
      'crop_n_layers': 1,
      'crop_n_points_downscale_factor': 2,
      'min_mask_region_area': 100,
    }
  else:
    kwargs = {}

  amg = SamAutomaticMaskGenerator(sam, **kwargs)
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


if __name__ == '__main__':
  parser = get_parser()
  parser.add_argument('-R', '--runner', default='amg', choices=['sam', 'ptor', 'amg'])
  args = get_args(parser)

  img = load_img(args.f)  # [H, W, C]
  sam = load_sam(args.M)
  
  globals()[f'run_{args.runner}'](args, sam, img)
