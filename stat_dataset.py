#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/10/23

from utils import *
from traceback import print_exc

from tqdm import tqdm
import matplotlib.pyplot as plt


number = Union[int, float]


def make_sam(args):
  from atk_sam import DATA_ROOT as SAM_DATA_ROOT
  sample_ids = sorted({fp.stem for fp in SAM_DATA_ROOT.iterdir()})

  area_dict: Dict[int, List[int]] = {}
  total_area_dict: Dict[int, int] = {}
  for id in tqdm(sample_ids):
    cfg = load_cfg(SAM_DATA_ROOT / f'{id}.json')
    cfg_img = cfg['image']
    H, W = cfg_img['height'], cfg_img['width']
    img_id = cfg_img['image_id']

    total_area_dict[img_id] = H * W
    area_dict[img_id]: List[int] = []
    for annot in cfg['annotations']:
      area_dict[img_id].append(annot['area'])

  save_stats(args, area_dict, total_area_dict)

def save_stats(args, area_dict:Dict[int, List[int]], total_area_dict:Dict[int, int]):
  area_list = []
  ratio_list = []
  for k, v in area_dict.items():
    area_list.extend(v)
    ratio_list.extend(np.asarray(v) / total_area_dict[k])
  area: ndarray = np.asarray(area_list, dtype=np.int32)
  area_log = np.log(area)
  ratio: ndarray = np.asarray(ratio_list, dtype=np.float32)
  ratio_log = np.log(ratio)

  print('n_samples:', len(area_dict))
  print('n_masks:',   len(area_list))

  def mk_sect(x:ndarray):
    return {
      'max': x.max() .item(),
      'min': x.min() .item(),
      'avg': x.mean().item(),
      'std': x.std() .item(),
      'var': x.var() .item(),
    }

  data = {
    'n_samples':  len(area_dict),
    'n_masks':    len(area_list),
    'stat': {
      'area':  mk_sect(area),
      'ratio': mk_sect(ratio),
    },
    'area':       area_dict,
    'total_area': total_area_dict,
  }
  save_json(data, OUT_PATH / f'stat_{args.D}.json')

  plt.figure(figsize=(12, 6))
  plt.clf()
  plt.subplot(121) ; plt.hist(area_log,  bins=50) ; plt.title('log(area)')
  plt.subplot(122) ; plt.hist(ratio_log, bins=50) ; plt.title('log(area_ratio)')
  plt.savefig(OUT_PATH / f'stat_{args.D}.png', dpi=600)


def query(args):
  fp = OUT_PATH / f'stat_{args.D}.json'
  if not fp.exists(): globals()[f'make_{args.D}'](args)

  data = load_json(fp)
  area_dict: Dict[int, List[int]] = data['area']
  total_area_dict: Dict[int, int] = data['total_area']
  n_images: int = data['n_samples']
  n_annots: int = data['n_masks']

  print('=' * 76)
  print(f'  query how many masks in the dataset [{args.D}] satisfies condition, e.g.:')
  print('       10000 30000    area in in range [10000, 30000]')
  print('       0.01 0.03      area ratio in range [0.01, 0.03]')
  print('=' * 76)
  print('>> total_images:', n_images)
  print('>> total_annots:', n_annots)
  print('=' * 76)

  def query_count(kind:str, vmin:number, vmax:number) -> Tuple[int, int]:
    n_img, n_ant = 0, 0
    for k, v in area_dict.items():
      v = np.asarray(v)
      if kind == 'ratio':
        total_area = total_area_dict[k]
        v = np.asarray(v) / total_area

      cnt = ((vmin <= v) & (v <= vmax)).sum()
      if cnt > 0:
        n_img += 1
        n_ant += cnt

    return n_img, n_ant

  def parse_input(s:str) -> Union[str, Tuple[number, number]]:
    svals = [e.strip() for e in s.split(' ')]
    
    vals = [int(s) if s.isdigit() else float(s) for s in svals]
    assert len({type(v) for v in vals}) == 1
    kind = 'area' if type(vals[0]) == int else 'ratio'
    return kind, vals

  while True:
    s: str = input('>> input area(int) or ratio(float): ')
    try:
      kind, (vmin, vmax) = parse_input(s)
    except:
      print_exc()
      continue

    n_img, n_ant = query_count(kind, vmin, vmax)
    print(f'>> filtered {n_img} ({n_img / n_images:.3%}) images, {n_ant} ({n_ant / n_annots:.3%}) annots')


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-D', default='sam', choices=['sam'], help='dataset')
  args = parser.parse_args()

  try:
    query(args)
  except KeyboardInterrupt:
    print('Exit by Ctrl+C')
  except:
    print_exc()
