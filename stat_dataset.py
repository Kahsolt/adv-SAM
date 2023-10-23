#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/10/23

from atk_sam import *
from atk_sam import DATA_ROOT as SAM_DATA_ROOT
from atk_kitti import *
from atk_kitti import DATA_ROOT as KITTI_DATA_ROOT
from utils import *

from tqdm import tqdm
import matplotlib.pyplot as plt


def make_sam(args):
  sample_ids = sorted({fp.stem for fp in SAM_DATA_ROOT.iterdir()})

  area: List[int] = []
  ratio: List[float] = []
  for id in tqdm(sample_ids):
    cfg = load_cfg(SAM_DATA_ROOT / f'{id}.json')
    cfg_img = cfg['image']
    H, W = cfg_img['height'], cfg_img['width']
    total_area = H * W

    for annot in cfg['annotations']:
      area .append(annot['area'])
      ratio.append(annot['area'] / total_area)

  save_stats(args, len(sample_ids), area, ratio)

def make_kitti(args):
  sample_ids = sorted({fp.stem for fp in (KITTI_DATA_ROOT / 'image_2').iterdir()})

  area: List[int] = []
  ratio: List[float] = []
  for id in tqdm(sample_ids):
    annot = load_annot(KITTI_DATA_ROOT / 'instance' / f'{id}.png')
    H, W = annot.shape
    total_area = H * W

    for oid in sorted(set(annot.flat)):
      x: int = (annot == oid).sum()
      area.append(x)
      ratio.append(x / total_area)

  save_stats(args, len(sample_ids), area, ratio)

def save_stats(args, n_samples, area, ratio):
  area: ndarray = np.asarray(area, dtype=np.int32)
  area_log = np.log(area)
  ratio: ndarray = np.asarray(ratio, dtype=np.float32)
  ratio_log = np.log(ratio + 1e-8)

  print('n_samples:', n_samples)
  print('n_masks:',   len(area))

  def mk_sect(x:ndarray):
    return {
      'max': x.max() .item(),
      'min': x.min() .item(),
      'avg': x.mean().item(),
      'std': x.std() .item(),
      'var': x.var() .item(),
      'val': x.tolist(),
    }

  data = {
    'n_samples': n_samples,
    'n_masks':   len(area),
    'area':      mk_sect(area),
    'ratio':     mk_sect(ratio),
  }
  save_json(data, OUT_PATH / f'stat_{args.D}.json')

  plt.figure(figsize=(12, 6))
  plt.clf()
  plt.subplot(121) ; plt.hist(area_log,  bins=50) ; plt.title('log(area)')
  plt.subplot(122) ; plt.hist(ratio_log, bins=50) ; plt.title('log(area_ratio)')
  plt.savefig(OUT_PATH / f'stat_{args.D}.png', dpi=600)


def query(args):
  data = load_cfg(OUT_PATH / f'stat_{args.D}.json')
  area  = np.asarray(data['area' ]['val'], dtype=np.int32)
  ratio = np.asarray(data['ratio']['val'], dtype=np.float32)

  print('=' * 76)
  print(f'  query how many masks in the dataset [{args.D}] satisfies condition, e.g.:')
  print('       10000       area >= 10000')
  print('       0.01 0.03   area ratio in range [0.01, 0.03]')
  print('=' * 76)

  number = Union[int, float]
  def parse_input(s:str) -> Union[Tuple[number, number], number]:
    if ' ' in s:
      svals = [e.strip() for e in s.split(' ')]
    else:
      svals = [s.strip()]
    
    vals = [int(s) if s.isdigit() else float(s) for s in svals]
    assert len({type(v) for v in vals}) == 1
    kind = 1 if type(vals[0]) == int else 0
    nparam = len(vals)
    assert nparam in [1, 2]
    return vals, kind, nparam

  while True:
    s: str = input('>> input area(int) or ratio(float): ')
    try:
      q, kind, nparam = parse_input(s)
    except:
      print_exc()
      continue

    if kind == 1:   # area
      if nparam == 1:
        ok = q <= area
      else:
        ok = (q[0] <= area) & (area <= q[1])
    else:   # area ratio
      if nparam == 1:
        ok = q <= ratio
      else:
        ok = (q[0] <= ratio) & (ratio <= q[1])

    print(f'>> found {sum(ok)} ({sum(ok) / len(ok):.3%}) masks')


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-D', default='sam', choices=['sam', 'kitti'], help='dataset')
  parser.add_argument('--make', action='store_true')
  parser.add_argument('-Q', '--query', action='store_true')
  args = parser.parse_args()

  assert args.make ^ args.query, 'must specify either --make or --query'

  if args.query:
    try:
      query(args)
    except KeyboardInterrupt:
      print('Exit by Ctrl+C')
    except:
      print_exc()
  else:
    globals()[f'make_{args.D}'](args)
