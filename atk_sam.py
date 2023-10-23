#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/08/03

from atk import *

from pycocotools.mask import decode

DATA_ROOT = DATASET_PATH['sam']


@timer
def run(args, sample_ids, runner, sampler=(lambda _, x: x)):
  sample_ids = sampler(args, sample_ids)
  if not sample_ids:
    print('>> warn: filtered no image samples to run :(')
    return

  sam = load_sam(args.M)
  ptor = SamPredictor(sam)
  fwder = SamForwarder(sam)
  loss_fn = make_loss_fn(args)

  HIST_FILE = OUT_PATH / f'atk_{args.D}.json'
  hist: List = load_json(HIST_FILE, list)
  s = time()

  iou_list, piou_list = [], []
  step_list = []
  interrupted = False
  try:
    runner(
      args,
      sample_ids,
      ptor, fwder, loss_fn,
      iou_list, piou_list, step_list,
    )
  except KeyboardInterrupt:
    print('>> interrupted!!')
    interrupted = True
  except:
    print_exc()
  finally:
    miou  = mean(iou_list)
    mpiou = mean(piou_list)
    mstep = mean(step_list)
    print(f'>> miou: {miou}')
    print(f'>> mpiou: {mpiou}')
    print(f'>> mstep: {mstep}')

    t = time()
    rec = {
      'n_image':     len(sample_ids),
      'n_annot':     len(iou_list),
      'miou':        miou,
      'mpiou':       mpiou,
      'mstep':       mstep,
      'iou_list':    iou_list,
      'piou_list':   piou_list,
      'step_list':   step_list,
      'interrupted': interrupted,
      'ts':          t - s,
      'ts_start':    str(datetime.fromtimestamp(t)),
      'ts_finish':   str(datetime.fromtimestamp(s)),
      'args':        vars(args),
    }
    hist.insert(0, rec)
    save_json(hist, HIST_FILE)

    if args.log:
      fp = args.log_dp / 'args.json'
      data = load_json(fp)
      for k, v in rec.items():
        if k in ['args']: continue
        if k not in data:
          data[k] = v
      save_json(data, fp)

def parser_filter(args) -> Tuple[str, Tuple[number, number]]:
  parse_cmd_arg = lambda s, cvt: [cvt(e) for e in s.split(',')]

  kind = None
  vmin, vmax = 0, 0
  if args.filter_area:
    kind = 'area'
    vmin, vmax = parse_cmd_arg(args.filter_area, int)
  if args.filter_ratio:
    kind = 'ratio'
    vmin, vmax = parse_cmd_arg(args.filter_ratio, float)
  assert vmin <= vmax
  return kind, (vmin, vmax)


def sample_sam_samples(args, sample_ids) -> List[Path]:
  np.random.shuffle(sample_ids)   # NOTE: this should be fixed by randseed
  if args.limit_img > 0: sample_ids = sample_ids[:args.limit_img]
  kind, (vmin, vmax) = parser_filter(args)

  def count_annots_by_filter(annots:List[dict], total_area:float) -> int:
    if kind is None: return len(annots)

    cnt = 0
    for annot in annots:
      v = annot['area']
      if kind == 'ratio':
        v /= total_area
      if vmin <= v <= vmax: cnt += 1
    return cnt

  sample_ids_filtered = []
  annots_filtered = 0
  for id in tqdm(sample_ids, desc='Image'):
    cfg = load_cfg(DATA_ROOT / f'{id}.json')
    total_area = cfg['image']['height'] * cfg['image']['width']
    cnt = count_annots_by_filter(cfg['annotations'], total_area)
    if not cnt: continue
    annots_filtered += cnt
    sample_ids_filtered.append(id)
    if args.limit_ant > 0 and annots_filtered >= args.limit_ant: break

  print(f'>> images_filtered: {len(sample_ids_filtered)}')
  print(f'>> annots_filtered: {annots_filtered}')
  return sample_ids_filtered

def run_sam_samples(args, sample_ids:List[Path], ptor:SamPredictor, fwder:SamForwarder, loss_fn:Callable, iou_list:List[int], piou_list:List[int], step_list:List[int]):
  kind, (vmin, vmax) = parser_filter(args)

  def filter_annots(annots:List[dict], total_area:float) -> List[dict]:
    if kind is None: return annots

    ret = []
    for annot in annots:
      v = annot['area']
      if kind == 'ratio':
        v /= total_area
      if vmin <= v <= vmax: ret.append(annot)
    return ret

  annots_processed = 0
  finished = False
  for id in tqdm(sample_ids, desc='Image'):
    if finished: break

    img = load_img(DATA_ROOT / f'{id}.jpg')
    img_size = img.shape[:-1]
    cfg = load_cfg(DATA_ROOT / f'{id}.json')
    total_area = cfg['image']['height'] * cfg['image']['width']
    annots = filter_annots(cfg['annotations'], total_area)

    if args.atk and args.tgt and len(annots) < 2: continue

    for annot in tqdm(annots, desc='Annot'):
      if finished: break

      if args.log:
        log_dp: Path = args.log_dp / id / str(annot['id'])
        log_dp.mkdir(parents=True, exist_ok=True)
      else:
        log_dp = None

      if 'input':
        point = np.asarray(annot['point_coords'])
        prompts = make_prompts(point, img_size)
      if 'ground truth':
        mask_gt: npimg_b1 = np.ascontiguousarray(decode(annot['segmentation']), dtype=bool)
        #piou_gt: float = annot['predicted_iou']
      
      fwd_pack = fwder, prompts, loss_fn
      ptor_pack = ptor, prompts

      if not args.atk or args.log:
        mask_img, piou_img = make_pred(ptor_pack, img, multi_mask=args.multi_mask)
        mask_hat, piou_hat = mask_img, piou_img

      if args.atk:
        if args.tgt:
          annots_tgt = annot
          while annots_tgt is annot: annots_tgt = np.random.choice(annots, size=1, replace=False)[0]
          point_tgt = np.asarray(annots_tgt['point_coords'])
          tgt = make_tgt(ptor, img, point_tgt)
        else:
          tgt = None

        adv, mask_adv, piou_adv, steps = pgd(args, fwd_pack, img, tgt, multi_mask=args.multi_mask, verbose=args.verbose, log_dp=log_dp)
        mask_hat, piou_hat = mask_adv, piou_adv
        step_list.append(steps)

        if args.log:
          plot6(img, mask_img, piou_img, adv, mask_adv, piou_adv, prompts, tgt, log_dp / 'plot6.png')

      if args.log:
        plot3(annot['point_coords'][0], img, mask_hat, mask_gt, log_dp / 'plot3.png')

      iou_list.append(get_iou_auto(mask_hat, mask_gt))
      piou_list.append(piou_hat)

      annots_processed += 1
      if args.limit_ant > 0 and annots_processed >= args.limit_ant:
        finished = True
        break


def get_parser() -> ArgumentParser:
  from atk import get_parser as get_base_parser

  parser = get_base_parser()
  # limit run sample count
  parser.add_argument('-L', '--limit_img', default=-1, type=int, help='limit run image count, set -1 for no limit')
  parser.add_argument('-K', '--limit_ant', default=-1, type=int, help='limit run annot count, set -1 for no limit')
  parser.add_argument('--filter_area',  type=str, help='filter sample by area, e.g.: 3000,5000')
  parser.add_argument('--filter_ratio', type=str, help='filter sample by area ratio, e.g.: 0.03,0.05')
  # attack settings
  parser.add_argument('--atk', action='store_true', help='enable PGD attack')
  parser.add_argument('--tgt', action='store_true', help='enable targeted attack (use randomly another mask as the target)')
  parser.add_argument('--multi_mask', action='store_true', help='use essay method to calc mIoU (pick the highest IoU from multipile mask outputs)')
  parser.add_argument('--verbose', action='store_true', help='show PGD progress & dist metrics')
  return parser

def get_args(parser:ArgumentParser) -> Namespace:
  from atk import get_args as get_base_args

  args = get_base_args(parser)
  args.f = None
  args.D = 'sam'

  if args.log: assert not args.multi_mask, 'setting conflict, cannot set both --log and --multi_mask'
  if args.filter_area and args.filter_ratio: 'setting conflict, cannot set both --filter_area and --filter_ratio'
  return args


if __name__ == '__main__':
  parser = get_parser()
  args = get_args(parser)
  mk_log(args)

  sample_ids = sorted({fp.stem for fp in DATA_ROOT.iterdir()})
  run(args, sample_ids, run_sam_samples, sample_sam_samples)
