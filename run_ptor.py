#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/06/26 

from utils import *


def run(args):
  img = load_img(args.f)  # [H, W, C]
  sam = load_sam(args.M)

  predictor = SamPredictor(sam)
  predictor.set_image(img)

  if 'point a coord':
    coords = np.asarray([[200, 400]])
    labels = np.asarray([1])
  masks, iou_predictions, low_res_masks = predictor.predict(point_coords=coords, point_labels=labels, multimask_output=False)

  for mask, piou, lmask in zip(masks, iou_predictions, low_res_masks):
    print('piou:', piou.item())

    plt.clf()
    plt.subplot(121) ; sns.heatmap( mask) ; plt.title('mask')
    plt.subplot(122) ; sns.heatmap(lmask) ; plt.title('mask orig.')
    plt.suptitle(f'piou: {piou.item()}')
    plt.show()


if __name__ == '__main__':
  run(get_args())
