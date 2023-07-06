#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/06/23 

import tkinter as tk
import tkinter.ttk as ttk
import tkinter.messagebox as tkmsg
import tkinter.filedialog as tkfdlg
from PIL import Image
from PIL.ImageTk import PhotoImage
from traceback import print_exc

from run_sam import *

WINDOW_TITLE = 'SAM interactive'
WINDOW_SIZE  = (700, 600)
RED_SHIFT    = 35


class App:

  def __init__(self, args:Namespace):
    self.args: Namespace = args
    self.cur_model: str = None
    self.predictor: SamPredictor = None
    self.img: npimgi = None
    self.img_red: npimgi = None
    self.img_grey: npimgi = None

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
    wnd.resizable(False, False)
    wnd.title(WINDOW_TITLE)
    wnd.protocol('WM_DELETE_WINDOW', wnd.quit)
    self.wnd = wnd

    # top: control
    frm1 = ttk.Frame(wnd)
    frm1.pack(side=tk.TOP, anchor=tk.N, expand=tk.YES, fill=tk.X)
    if True:
      self.var_model = tk.StringVar(frm1, value=self.args.M)
      cb = ttk.Combobox(frm1, state='readonly', values=list(SAM_CKPTS.keys()), textvariable=self.var_model, width=12)
      cb.bind('<<ComboboxSelected>>', lambda evt: self.change_model())
      cb.pack(side=tk.LEFT)
    
      self.var_fp = tk.StringVar(wnd, value=self.args.f)
      tk.Entry(frm1, textvariable=self.var_fp).pack(side=tk.LEFT, expand=tk.YES, fill=tk.X)
      tk.Button(frm1, text='Open..', command=self._open).pack(side=tk.RIGHT)

    # bottom: display
    frm2 = ttk.Frame(wnd)
    frm2.pack(anchor=tk.CENTER, expand=tk.YES, fill=tk.BOTH)
    if True:
      pv = ttk.Label(frm2)
      pv.bind('<Button-1>', lambda evt: self.infer(evt))
      pv.bind('<Button-3>', lambda evt: self._show())
      pv.pack(side=tk.LEFT, fill=tk.X, expand=tk.YES)
      self.pv = pv

  def _open(self):
    fp = tkfdlg.askopenfilename()
    if not fp: return
    if not Path(fp).is_file():
      tkmsg.showerror('Error', f'path {fp} is not a file!')
      return
    
    self.var_fp.set(fp)
    self.change_image()

  def _show(self, img=None):
    img = img if img is not None else self.img
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
    
    self.img = np.asarray(load_img(fp), dtype=np.uint8)
    print(f'[image] size {self.img.shape[:-1]}')
    self._show()
    self.predictor.set_image(self.img)
    
    if 'overlays':
      self.img_red  = img_to_red (self.img, RED_SHIFT)
      self.img_grey = img_to_grey(self.img)

  def infer(self, evt:tk.Event):
    if self.predictor is None:
      tkmsg.showerror('Error', 'no model loaded')
      return
    if self.img is None:
      tkmsg.showerror('Error', 'no image loaded')
      return

    sel_y, sel_x = evt.y, evt.x
    coords = np.asarray([[sel_x, sel_y]])
    labels = np.asarray([1])
    print(f'<< point: ({sel_y}, {sel_x})')

    masks, iou_predictions, low_res_logits = self.predictor.predict(
      point_coords=coords, 
      point_labels=labels, 
      multimask_output=False,
      return_logits=True,
    )
    masks_bin = masks > self.predictor.model.mask_threshold

    masks_bin            # [B=1, H=534, W=800], bool
    masks                # [B=1, H=534, W=800], float
    low_res_logits       # [B=1, H=256, W=256], float
    iou_predictions      # [B=1, N=1], float
    print(f'>> iou: {iou_predictions.item()}')

    mask = np.expand_dims(masks_bin[0], -1)   # [H, W, C=1]
    seg = self.img_red * mask + self.img_grey * ~mask
    self._show(seg)


if __name__ == '__main__':
  App(get_args())
