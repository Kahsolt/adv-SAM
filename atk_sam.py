#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/08/03

from atk import *

try:
  from pycocotools.mask import decode
except ImportError:
  print('>> [error] missing lib "pycocotools", run "pip install pycocotools" first!!')
  raise

DATA_ROOT = DATASET_PATH['sam']
HIST_FILE = OUT_PATH / 'atk_sam.json'


@timer
def run(args):
  sample_ids = sorted({fp.stem for fp in DATA_ROOT.iterdir()})
  np.random.shuffle(sample_ids)
  if args.limit_img > 0: sample_ids = sample_ids[:args.limit_img]
  
  sam = load_sam(args.M)
  ptor = SamPredictor(sam)
  fwder = SamForwarder(sam)
  loss_fn = make_loss_fn(args)

  hist: List = load_json(HIST_FILE, list)
  s = time()

  iou_sum, iou_cnt = 0.0, 0
  interrupted = False
  try:
    for id in tqdm(sample_ids):
      img = load_img(DATA_ROOT / f'{id}.jpg')
      img_size = img.shape[:-1]

      cfg = load_cfg(DATA_ROOT / f'{id}.json')
      annots = cfg['annotations']
      annots_sel = np.random.choice(annots, size=args.limit_ant, replace=False) if 0 < args.limit_ant < len(annots) else annots

      for annot in annots_sel:
        if 'input':
          point = np.asarray(annot['point_coords'])
          prompts = make_prompts(point, img_size)
        if 'ground truth':
          mask_gt: npimg_b1 = np.ascontiguousarray(decode(annot['segmentation']), dtype=bool)
          piou_gt: float = annot['predicted_iou']
        
        fwd_pack = fwder, prompts, loss_fn
        ptor_pack = ptor, prompts

        if args.atk:
          if args.tgt:
            annots_tgt = annot
            while annots_tgt is annot: annots_tgt = np.random.choice(annots, size=1, replace=False)[0]
            point_tgt = np.asarray(annots_tgt['point_coords'])
            tgt = make_tgt(ptor, img, point_tgt)
          else:
            tgt = None
        
          lim = make_lim(args, img, tgt, ptor_pack, fwd_pack)

          _, mask_hat, piou_hat = pgd(args, fwd_pack, img, tgt, lim, multi_mask=args.multi_mask, log=args.debug)
        else:
          mask_hat, piou_hat = make_pred(ptor_pack, img, multi_mask=args.multi_mask)

        iou_cnt += 1
        iou_sum += get_iou_auto(mask_hat, mask_gt)

  except KeyboardInterrupt:
    print('>> interrupted!!')
    interrupted = True
  except:
    print_exc()
  finally:
    miou = 0.0 if iou_cnt == 0 else (iou_sum / iou_cnt)
    print(f'>> miou: {miou}')

    t = time()
    rec = {
      'miou': miou,
      'interrupted': interrupted,
      'ts': t - s,
      'ts_start': str(datetime.fromtimestamp(t)),
      'ts_finish': str(datetime.fromtimestamp(s)),
      'args': vars(args),
    }
    hist.insert(0, rec)
    save_json(hist, HIST_FILE)


def get_parser() -> ArgumentParser:
  from atk import get_parser as get_base_parser

  parser = get_base_parser()
  parser.add_argument('-L', '--limit_img', default=-1, type=int, help='limit run image count, set -1 for all')
  parser.add_argument('-K', '--limit_ant', default=1,  type=int, help='limit run annot count of each image, set -1 for all')
  parser.add_argument('--atk', action='store_true', help='enable PGD attack')
  parser.add_argument('--tgt', action='store_true', help='enable targeted attack (use randomly another mask as the target)')
  parser.add_argument('--multi_mask', action='store_true', help='use essay method to calc mIoU (pick the highest IoU from multipile mask outputs)')
  return parser

def get_args(parser:ArgumentParser) -> Namespace:
  from atk import get_args as get_base_args

  args = get_base_args(parser)
  args.f = None
  args.D = 'sam'
  args.fps = -1
  args.debug = False
  return args


if __name__ == '__main__':
  parser = get_parser()
  args = get_args(parser)
  mk_log(args)
  run(args)
