#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/06/23 

from utils import *


def Sam_forward(self:Sam, batched_input:List[Dict[str, Any]]) -> List[Dict[str, torch.Tensor]]:
  input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)     # [B=1, C=3, H=1024, W=1024], float
  image_embeddings = self.image_encoder(input_images)     # [B=1, C=256, H=64, W=64], x16 downsample

  outputs = []
  for image_record, curr_embedding in zip(batched_input, image_embeddings):
      sparse_embeddings, dense_embeddings = self.prompt_encoder(    # [1, 0, 256], [1, 256, 64, 64]
        points=None,
        boxes=None,
        masks=None,
      )
      low_res_masks, iou_predictions = self.mask_decoder(   # [B=1, C=1, H=256, W=256], x4 upsample; [B=1, 1]
        image_embeddings=curr_embedding.unsqueeze(0),
        image_pe=self.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
      )
      masks = self.postprocess_masks(
          low_res_masks,
          input_size=image_record["image"].shape[-2:],
          original_size=image_record["original_size"],
      )
      masks = masks > self.mask_threshold
      outputs.append({
        "masks": masks,
        "iou_predictions": iou_predictions,
        "low_res_logits": low_res_masks,
      })
  return outputs


def run(args):
  img = load_img(args.f)  # [H, W, C]
  sam = load_sam(args.M)

  X = torch.from_numpy(img).permute([2, 0, 1])  # [0, 255]
  print('X.shape:', X.shape)

  inputs = [{
    'image': X.to(device),
    'original_size': X.shape[1:],
  }]
  ret = Sam_forward(sam, inputs)[0]
  ret['masks']                # [B=1, C=3, H=534, W=800], bool
  ret['low_res_logits']       # [B=1, C=3, H=256, W=256], float
  ret['iou_predictions']      # [B=1, D=3], float
  breakpoint()


if __name__ == '__main__':
  run(get_args())
