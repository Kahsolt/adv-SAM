#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/07/05 

from pathlib import Path
from PIL import Image, ImageFilter
from PIL.Image import Image as PILImage
from argparse import ArgumentParser, Namespace
from typing import *

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torchvision.utils import make_grid
import numpy as np
from numpy.typing import NDArray
import seaborn as sns
import matplotlib.pyplot as plt

if 'repo':
  BASE_PATH = Path(__file__).parent.absolute()
  REPO_PATH = BASE_PATH / 'repo'
  SAM_PATH  = REPO_PATH / 'segment-anything'
  SAM_CKPT_PATH = SAM_PATH / 'ckpt'
  SAM_DEMO_FILE = SAM_PATH / 'notebooks' / 'images' / 'dog.jpg'
  SAM_CKPTS = {
    'vit_b': 'sam_vit_b_01ec64.pth',
    'vit_l': 'sam_vit_l_0b3195.pth',
    'vit_h': 'sam_vit_h_4b8939.pth',
  }

  import sys
  sys.path.append(str(SAM_PATH))
  from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
  from segment_anything.modeling import Sam

device = 'cuda' if torch.cuda.is_available() else 'cpu'

npimg_u8  = NDArray[np.uint8]
npimg_f32 = NDArray[np.float32]
npimg_b1  = NDArray[np.bool_]
npimg     = Union[npimg_u8, npimg_f32, npimg_b1]
Size      = Tuple[int, int]
Point     = Tuple[int, int]

def load_img(fp:Path, mode='RGB', dtype=np.uint8) -> Tuple[npimg_u8, npimg_f32]:
  assert dtype in [np.uint8, np.float32], 'invalid dtype, should be in [np.uint8, np.float32]'

  img = Image.open(str(fp)).convert(mode)
  im = np.array(img, dtype=np.uint8)
  if dtype is np.uint8: return im
  return (im / 255.0).astype(np.float32)

def show_img(im:npimg, figsize=(8, 6), anns:Dict[str, Any]=None):
  plt.figure(figsize=figsize)
  plt.imshow(im)
  if anns: show_anns(anns)
  plt.axis('off')
  plt.show()

def show_anns(anns:Dict[str, Any]):
  '''
    anns: the direct output of SamAutomaticMaskGenerator.generate
  '''
  if len(anns) == 0: return

  sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
  img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
  img[:,:,3] = 0
  for ann in sorted_anns:
    m = ann['segmentation']
    color_mask = np.concatenate([np.random.random(3), [0.35]])
    img[m] = color_mask

  ax = plt.gca()
  ax.set_autoscale_on(False)
  ax.imshow(img)

def get_mask_edge(im:npimg_u8, thresh:float=0.5) -> npimg_b1:
  assert is_npimg_u8(im), 'expect npimg of np.uint8'
  img = Image.fromarray(im)
  img = img.convert('RGB').filter(ImageFilter.FIND_EDGES).convert('L')
  return np.asarray(img) > (thresh * 255.0)


def img_to_red(im:npimg_u8, shift:int=25) -> npimg_u8:
  im = np.copy(im).astype(np.uint16)
  im[:, :, 0] += shift
  im = im.clip(0, 255)
  return im.astype(np.uint8)

def img_to_grey(im:npimg_u8) -> npimg_u8:
  img = Image.fromarray(im).convert('L')
  im = np.asarray(img, dtype=np.uint8)
  return np.expand_dims(im, -1)   # [H, W, C=1]

def is_npimg_u8(im:npimg) -> bool:
  if not isinstance(im, np.ndarray): return False
  if im.dtype != np.uint8: return False
  if len(im.shape) not in [2, 3]: return False
  if len(im.shape) == 3 and im.shape[-1] not in [1, 3]: return False
  return True

def minmax_norm(x:np.ndarray) -> np.ndarray:
  return (x - x.min()) / (x.max() - x.min())

def info_t(x:Union[np.ndarray, Tensor], name:str='x'):
  print(f'{name}: shape={tuple(x.shape)}, dtype={x.dtype}')


def load_sam(model:str) -> Sam:
  fp = SAM_CKPT_PATH / SAM_CKPTS[model]
  print(f'>> load weights from {fp}')
  return sam_model_registry[model](checkpoint=str(fp)).eval().to(device)

def get_param_cnt(model:nn.Module) -> int:
  return sum([p.numel() for p in model.parameters() if p.requires_grad])


def get_parser() -> ArgumentParser:
  parser = ArgumentParser()
  parser.add_argument('-M', default='vit_b', choices=SAM_CKPTS.keys())
  parser.add_argument('-f', default=SAM_DEMO_FILE, type=Path)
  return parser

def get_args(parser:ArgumentParser=None) -> Namespace:
  parser = parser or get_parser()
  args = parser.parse_args()

  assert Path(args.f).is_file(), f'>> {args.f} is not a file'

  return args
