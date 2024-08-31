#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/06/23 

import tkinter as tk
import tkinter.ttk as ttk
import tkinter.messagebox as tkmsg
import tkinter.filedialog as tkfdlg
from PIL import Image
from PIL.ImageTk import PhotoImage
from inspect import signature
from traceback import print_exc

from run_cui import *

device = 'cpu'

WINDOW_TITLE = 'SAM interactive'
WINDOW_SIZE  = (800, 600)
RESIZE_HW    = 800
RED_SHIFT    = 35


def img_resize(img:PILImage) -> PILImage:
  w, h = img.size
  if max(w, h) <= RESIZE_HW:
    return img

  if w < h:
    nh = RESIZE_HW
    nw = round(w * nh / h)
  else:
    nw = RESIZE_HW
    nh = round(h * nw / w)
  return img.resize((nw, nh))


class App:

  def __init__(self, args:Namespace):
    self.args: Namespace = args
    self.cur_model: str = None
    self.predictor: SamPredictor = None
    self.img: npimg_u8 = None
    self.img_o: npimg_u8 = None

    self.setup_gui()
    self.init_workspace()

    try:
      self.wnd.mainloop()
    except KeyboardInterrupt:
      self.wnd.destroy()
    except: print_exc()

  def init_workspace(self):
    self.change_model()
    self.change_image()

  def setup_gui(self):
    # window
    wnd = tk.Tk()
    W, H = wnd.winfo_screenwidth(), wnd.winfo_screenheight()
    w, h = WINDOW_SIZE
    wnd.geometry(f'{w}x{h}+{(W-w)//2}+{(H-h)//2}')
    #wnd.resizable(False, False)
    wnd.title(WINDOW_TITLE)
    wnd.protocol('WM_DELETE_WINDOW', wnd.quit)
    self.wnd = wnd

    # top: control
    frm1 = ttk.Frame(wnd)
    frm1.pack(side=tk.TOP, anchor=tk.N, expand=tk.YES, fill=tk.X)
    if True:
      model_default = self.args.M if self.args.M in SAM_CKPTS else list(SAM_CKPTS.keys())[0]
      self.var_model = tk.StringVar(frm1, value=model_default)
      cb = ttk.Combobox(frm1, state='readonly', values=list(SAM_CKPTS.keys()), textvariable=self.var_model, width=12)
      cb.bind('<<ComboboxSelected>>', lambda evt: self.change_model())
      cb.pack(side=tk.LEFT)
    
      self.var_fp = tk.StringVar(wnd, value=self.args.f)
      tk.Entry(frm1, textvariable=self.var_fp).pack(side=tk.LEFT, expand=tk.YES, fill=tk.X)
      tk.Button(frm1, text='Open..', command=self._open).pack(side=tk.RIGHT)

    # mid: display
    frm2 = ttk.Frame(wnd)
    frm2.pack(side=tk.TOP, expand=tk.YES, fill=tk.BOTH)
    if True:
      pv = ttk.Label(frm2)
      pv.bind('<Button-1>', lambda evt: self.infer(evt))
      pv.bind('<Button-3>', lambda evt: self._show())
      pv.pack(side=tk.TOP, expand=tk.YES, fill=tk.BOTH)    # NOTE: must left-align for coord transform
      self.pv = pv

    # bottom: help
    frm3 = ttk.Frame(wnd)
    frm3.pack(side=tk.BOTTOM, anchor=tk.S, expand=tk.YES, fill=tk.X)
    if True:
      lbl = ttk.Label(frm3, text='>> Left click: put a pointer, Right click: clear the pointer')
      lbl.pack(side=tk.LEFT, fill=tk.X, expand=tk.YES)

  def _open(self):
    fp = tkfdlg.askopenfilename()
    if not fp: return
    if not Path(fp).is_file():
      tkmsg.showerror('Error', f'path {fp} is not a file!')
      return
    
    self.var_fp.set(fp)
    self.change_image()

  def _show(self, img=None):
    img = img if img is not None else self.img_r
    img = PhotoImage(Image.fromarray(img))
    self.pv.configure(image=img)
    self.pv.img = img

  def change_model(self):
    name = self.var_model.get()
    if name == self.cur_model: return

    model = load_sam(name).to(device)
    self.predictor = SamPredictor(model)
    self.cur_model = name
    print(f'[model] load model {name!r}, param_cnt: {get_param_cnt(model)}')

  def change_image(self):
    fp = self.var_fp.get()
    if not fp: return
    if not Path(fp).is_file(): return
    print(f'[image] load image from {fp!s}')
    
    img = Image.open(fp).convert('RGB')
    img_r = img_resize(img)
    print(f'[image] size {img.size} => resized: {img_r.size}')
    self.img   = np.asarray(img,   dtype=np.uint8)  # infer
    self.img_r = np.asarray(img_r, dtype=np.uint8)  # display
    self.img_o = img_to_red(self.img_r, RED_SHIFT)  # display
    self._show()
    self.predictor.set_image(self.img)

  def infer(self, evt:tk.Event):
    if self.predictor is None:
      tkmsg.showerror('Error', 'no model loaded')
      return
    if self.img is None:
      tkmsg.showerror('Error', 'no image loaded')
      return

    sel_y, sel_x = evt.y, evt.x
    if sel_y > self.img_r.shape[0] or sel_x > self.img_r.shape[1]: return

    pt_y = sel_y / self.img_r.shape[0] * self.img.shape[0]
    pt_x = sel_x / self.img_r.shape[1] * self.img.shape[1]
    coords = np.asarray([[pt_x, pt_y]])   # (h, w)
    labels = np.asarray([1])
    print(f'<< cursor: ({sel_y}, {sel_x}), point: ({pt_y}, {pt_x})')

    predict_kwargs = {
      'point_coords': coords,
      'point_labels': labels,
      'multimask_output': args.multi_mask,
      'return_logits': True,
    }
    if 'multimask_output' not in signature(self.predictor.predict).parameters:  # compact for TinySAM
      del predict_kwargs['multimask_output']
    masks, iou_predictions, low_res_logits = self.predictor.predict(**predict_kwargs)

    if args.multi_mask:   # select the largest piou
      idx = np.argmax(iou_predictions)
      iou_predictions = iou_predictions[idx]
      masks = masks[idx:idx+1]
    masks_bin = masks > self.predictor.model.mask_threshold

    B, H, W = masks.shape
    total_area = H * W
    area = masks_bin.sum()
    print(f'>> area: {area} ({area / total_area:.5%}), piou: {iou_predictions.item()}')

    if not 'debug':
      masks_bin            # [B=1, H=534, W=800], bool
      masks                # [B=1, H=534, W=800], float
      low_res_logits       # [B=1, H=256, W=256], float
      iou_predictions      # [B=1, N=1], float

    if 'unresize':
      img = Image.fromarray(masks_bin[0]).convert('L')
      img = img_resize(img)
      masks_bin = np.expand_dims(np.asarray(img, dtype=bool), axis=0)

    mask = np.expand_dims(masks_bin[0], -1)   # [H, W, C=1]
    seg = self.img_o * mask + self.img_r * ~mask
    self._show(seg)


if __name__ == '__main__':
  parser = get_parser()
  parser.add_argument('--multi_mask', action='store_true', help='enable multi_mask output')
  args = get_args(parser)
  App(args)
