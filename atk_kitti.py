#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/08/07

from queue import Queue

from atk import *
from atk_sam import get_parser as get_base_parser, get_args

DATA_ROOT = DATASET_PATH['kitti'] / 'training'
HIST_FILE = BASE_PATH / 'atk_kitti.json'

# annot IDs ref: https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py

npimg_u16 = NDArray[np.uint16]

def load_annot(fp:Path) -> npimg_u16:
  img = Image.open(str(fp))
  annot = np.array(img, dtype=np.uint16)
  # semn: class id; inst: object id (within its class)
  #inst = annot  % 256
  #semn = annot // 256
  return annot

def image_clip(img:npimg_u8, annot:npimg_u16) -> Tuple[npimg_u8, npimg_u16]:
  ''' original aspect ratio is about 1:3.3 (hw), we split it by 1/3 to get about 1:1 '''

  H, W = annot.shape
  hW = W // 3
  r = random.randint(0, W - hW - 1)
  slicer = slice(r, r + hW)
  return img[:, slicer, :], annot[:, slicer]

def mask_pick_connected(mask:npimg_b1, thresh:float=0.0) -> npimg_b1:
  ''' original semantical mask could be not connected, we randomly pick a connected area '''

  H, W = mask.shape
  min_area = int(mask.sum() * thresh) if thresh <= 1.0 else int(thresh)

  pts = np.stack(np.where(mask), axis=-1)   # [N, D=2]
  pool = list(range(len(pts)))
  random.shuffle(pool)

  visit = np.zeros_like(mask, dtype=np.int32)
  dx = [+1, +1, -1, -1, 0, 0, +1, -1]
  dy = [+1, -1, +1, -1, +1, -1, 0, 0]
  ndir = len(dx)
  q = Queue()
  def bfs(x:int, y:int, v:int):
    visit[x, y] = v
    q.put((x, y))
    while not q.empty():
      x, y = q.get()
      for d in range(ndir):
        nx = x + dx[d]
        ny = y + dy[d]
        if nx <  0: continue
        if nx >= H: continue
        if ny <  0: continue
        if ny >= W: continue
        if not mask[nx, ny]: continue
        if visit[nx, ny] > 0: continue
        visit[nx, ny] = v
        q.put((nx, ny))

  mid = 0
  def new_flood(x:int, y:float) -> npimg_b1:
    nonlocal mid
    mid += 1
    bfs(x, y, mid)
    return visit == mid

  submask = None
  cur_area = 0
  while cur_area < min_area and len(pool):
    x, y = pts[pool.pop()].tolist()
    if visit[x, y] > 0: continue

    submask = new_flood(x, y)
    cur_area = submask.sum()

  return submask if submask is not None else mask

def mask_centroid(mask:npimg_b1) -> Point:
  H, W = mask.shape
  area = mask.sum()

  sum_W = mask.sum(axis=1)
  sum = 0.0
  for x in range(H):
    sum += x * sum_W[x]
  x = sum / area

  sum_H = mask.sum(axis=0)
  sum = 0.0
  for y in range(W):
    sum += y * sum_H[y]
  y = sum / area
  
  if mask[int(x), int(y)]: return [x, y][::-1]

  pts = np.stack(np.where(mask), axis=-1)           # [N, D=2]
  pt = np.expand_dims(np.asarray([x, y]), axis=0)   # [1, D=2]
  dist2 = (pts - pt) ** 2
  return pts[dist2.argmin()].tolist()[::-1]


@timer
def run(args):
  sample_ids = sorted({fp.stem for fp in (DATA_ROOT / 'image_2').iterdir()})
  np.random.shuffle(sample_ids)
  if args.limit_img > 0: sample_ids = sample_ids[:args.limit_img]
  
  sam = load_sam(args.M)
  ptor = SamPredictor(sam)
  fwder = SamForwarder(sam)

  hist: List = load_json(HIST_FILE, list)
  s = time()

  iou_sum, iou_cnt = 0.0, 0
  interrupted = False
  try:
    for id in tqdm(sample_ids):
      img   = load_img  (DATA_ROOT / 'image_2'  / f'{id}.png')
      annot = load_annot(DATA_ROOT / 'instance' / f'{id}.png')
      img, annot = image_clip(img, annot)
      img_size = img.shape[:-1]
      
      oids = sorted(set(annot.flat))
      oids_sel = np.random.choice(oids, size=args.limit_ant, replace=False) if 0 < args.limit_ant < len(oids) else oids
      
      for oid in oids_sel:
        try:
          if 'ground truth':
            mask_gt: npimg_b1 = annot == oid
            mask_gt = mask_pick_connected(mask_gt, args.area_thresh)
          if 'input':
            xy = mask_centroid(mask_gt)
            point = np.asarray([xy], dtype=np.float32)
            prompts = make_prompts(point, img_size)

          if args.atk:
            if args.tgt:
              oid_tgt = np.random.choice(oids, size=1, replace=False)[0]
              while oid_tgt == oid: oid_tgt = np.random.choice(oids, size=1, replace=False)[0]
              point_tgt = np.asarray([mask_centroid(annot == oid_tgt)], dtype=np.float32)
              prompts_tgt = make_prompts(point_tgt, img_size)
              tgt, _ = make_pred(ptor, img, prompts_tgt)
            else:
              tgt = None
          
            if args.lim == 'edge': lim = get_mask_edge(img, args.thresh)
            if args.lim == 'tgt':  lim = tgt
            else:                  lim = None

            _, mask_hat, piou = pgd(args, fwder, prompts, img, tgt, lim, log=args.debug)
          else:
            mask_hat, piou = make_pred(ptor, img, prompts, multi_mask=args.multi_mask)

          if not 'debug':
            plt.figure(figsize=(6, 3), dpi=240)
            plt.subplot(131) ; plt.title('img')  ; plt.axis('off') ; plt.imshow(img)
            plt.text(*xy, s='★', color='r')
            plt.subplot(132) ; plt.title('pred') ; plt.axis('off') ; plt.imshow(mask_hat)
            plt.subplot(133) ; plt.title('GT')   ; plt.axis('off') ; plt.imshow(mask_gt)
            plt.tight_layout()
            plt.show()

          iou_cnt += 1
          iou_sum += get_iou_auto(mask_hat, mask_gt)
        except:
          print_exc()

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
  parser = get_base_parser()
  parser.add_argument('--area_thresh', default=0.05, type=float, help='minimal mask area to pick in percentage (<= 1.0) or absolute (> 1)')
  return parser


if __name__ == '__main__':
  run(get_args(get_parser()))
