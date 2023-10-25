#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/10/24

import os
import gc
import psutil
import math
import tkinter as tk
import tkinter.ttk as ttk
import tkinter.filedialog as tkfdlg
from traceback import print_exc

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from atk_sam import *
from atk_sam import DATA_ROOT as SAM_DATA_ROOT

__version__ = '0.1'

number = Union[float, int]
VRng = Tuple[number, number]

LIST_HEIGHT = 24
RESOLUTION: Dict[str, number] = {
  'area':  1000,
  'ratio': 0.001,
}


class App:

  def __init__(self):
    self.p = psutil.Process(os.getpid())

    self.cur_kind: str = None     # cur kind
    self.cur_annots_ids = set()   # cur filtered mask idx, to avoid unneccesary refresh
    self.last_vrng: Dict[str, Dict[str, VRng]] = {}   # id => kind => (vmin, vmax)
    self.cache: Dict[str, dict] = {}    # id => cfg

    self.setup_gui()
    self.setup_workspace()

    try:
      self.wnd.mainloop()
    except KeyboardInterrupt:
      self.wnd.quit()
    except: print_exc()

  def setup_gui(self):
    # window
    wnd = tk.Tk()
    wnd.title(f'Annotation Viewer {__version__}')
    wnd.protocol('WM_DELETE_WINDOW', wnd.quit)
    self.wnd = wnd

    # menu
    menu = tk.Menu(wnd, tearoff=0)
    menu.add_command(label='Save figure', command=self.save_fig)
    menu.add_separator()
    menu.add_command(label='Memory status', command=self.mem_stats)
    menu.add_command(label='Memory clean',  command=self.mem_clean)
    def menu_show(evt):
      try:     menu.tk_popup(evt.x_root, evt.y_root)
      finally: menu.grab_release()
    wnd.bind('<Button-3>', menu_show)

    # body
    frm0 = ttk.Label(wnd)
    frm0.pack(expand=tk.YES, fill=tk.BOTH)
    if True:
      # left: control
      frm1 = ttk.Frame(frm0)
      frm1.pack(side=tk.LEFT, expand=tk.YES, fill=tk.Y)
      if True:
        # top: action
        frm11 = ttk.LabelFrame(frm1, text='Settings')
        frm11.pack(side=tk.TOP, expand=tk.YES, fill=tk.X)
        if True:
          self.var_kind = tk.StringVar(wnd, value='area')
          self.var_vmax = tk.DoubleVar(wnd, value=1.0)
          self.var_vmin = tk.DoubleVar(wnd, value=0.0)

          frm111 = ttk.LabelFrame(frm11, text='Type')
          frm111.pack(expand=tk.YES, fill=tk.X)
          if True:
            cb = ttk.Combobox(frm111, text='type', values=['area', 'ratio'], textvariable=self.var_kind, state='readonly')
            cb.grid(row=0, column=1, padx=2)
            cb.bind('<<ComboboxSelected>>', lambda evt: self._change_type())

          frm112 = ttk.LabelFrame(frm11, text='Vmin')
          frm112.pack(expand=tk.YES, fill=tk.X)
          if True:
            sc = tk.Scale(frm112, command=lambda _: self.redraw(), variable=self.var_vmin, orient=tk.HORIZONTAL)
            sc.bind('<MouseWheel>', lambda evt: self._scroll_vrng('min', evt))
            sc.pack(expand=tk.YES, fill=tk.X)
            self.sc_min = sc

          frm113 = ttk.LabelFrame(frm11, text='Vmax')
          frm113.pack(expand=tk.YES, fill=tk.X)
          if True:
            sc = tk.Scale(frm113, command=lambda _: self.redraw(), variable=self.var_vmax, orient=tk.HORIZONTAL)
            sc.bind('<MouseWheel>', lambda evt: self._scroll_vrng('max', evt))
            sc.pack(expand=tk.YES, fill=tk.X)
            self.sc_max = sc

        # mid: info
        frm13 = ttk.Frame(frm1)
        frm13.pack(expand=tk.YES, fill=tk.X)
        if True:
          self.var_info = tk.StringVar(wnd)
          ttk.Label(frm13, textvariable=self.var_info).pack()

        # bottom: file list
        frm12 = ttk.LabelFrame(frm1, text='Images')
        frm12.pack(side=tk.BOTTOM, expand=tk.YES, fill=tk.BOTH)
        if True:
          self.var_ids = tk.StringVar()
          ls = tk.Listbox(frm12, listvariable=self.var_ids, selectmode=tk.BROWSE, height=LIST_HEIGHT)
          ls.bind('<<ListboxSelect>>', lambda evt: self._change_file())
          ls.pack(expand=tk.YES, fill=tk.BOTH)
          self.ls = ls

      # right: plot
      frm2 = ttk.LabelFrame(frm0, text='Annots')
      frm2.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.BOTH)
      if True:
        fig, ax = plt.subplots()
        fig.tight_layout()
        cvs = FigureCanvasTkAgg(fig, frm2)
        cvs.get_tk_widget().pack(expand=tk.YES, fill=tk.BOTH)
        self.fig, self.ax, self.cvs = fig, ax, cvs

    # status
    frm3 = ttk.Label(wnd)
    frm3.pack(side=tk.BOTTOM, anchor=tk.S, expand=tk.YES, fill=tk.X)
    if True:
      self.var_status = tk.StringVar(wnd)
      tk.Label(frm3, textvariable=self.var_status).pack(anchor=tk.W)

  def setup_workspace(self):
    self.ls.selection_clear(0, tk.END)
    self.var_ids.set([fp.stem for fp in SAM_DATA_ROOT.iterdir() if fp.suffix == '.json'])

    self.mem_clean()

  def _scroll_vrng(self, what, evt):
    try: id = self.ls.selection_get()
    except: return
    if not id: return
    if id not in self.cache: return
    cfg = self.cache[id]
    kind = self.var_kind.get()

    offset = 1 if evt.delta < 0 else -1
    vmin, vmax = cfg['vrng'][kind]
    var: tk.DoubleVar = getattr(self, f'var_v{what}')
    v = var.get() + offset * RESOLUTION[kind]
    v = max(min(v, vmax), vmin)   # clip to [vmin, vmax]
    var.set(v)

    self.redraw()

  def _change_type(self):
    try: id = self.ls.selection_get()
    except: return
    if not id: return
    if id not in self.cache: return
    kind = self.var_kind.get()

    if kind == self.cur_kind: return
    old_kind = self.cur_kind
    self.cur_kind = kind

    # save cfg
    if old_kind:
      self.last_vrng[id][old_kind] = (self.var_vmin.get(), self.var_vmax.get())

    # load cfg
    vrng = self.last_vrng[id][kind]
    self.var_vmin.set(vrng[0])
    self.var_vmax.set(vrng[1])

    self.redraw()

  def _change_file(self):
    try: id = self.ls.selection_get()
    except: return
    if not id: return
    if id in ['area', 'ratio']: return   # unknown dispatch bug

    # 加载文件
    if id not in self.cache:
      cfg = load_cfg(DATA_ROOT / f'{id}.json')
      H, W = cfg['image']['height'], cfg['image']['width']
      total_area = H * W

      # 预处理
      cfg['size'] = (H, W)
      area_max, area_min = 0, 1e10
      for annot in cfg['annotations']:
        area = annot['area']
        area_max = max(area, area_max)
        area_min = min(area, area_min)
        annot['ratio'] = area / total_area
        annot['mask']  = annot['segmentation']    # lazy decode
        annot['color'] = (np.concatenate([np.random.random(3), [0.35]]) * 255).astype(np.uint8)
        del_keys = [
          'segmentation',
          'point_coords',
          'bbox',
          'crop_box',
          'predicted_iou',
          'stability_score',
        ]
        for k in del_keys:
          del annot[k]
      cfg['vrng'] = {
        'area':  (area_min, area_max),
        'ratio': (area_min / total_area, area_max / total_area),
      }

      self.cache[id] = cfg
      self.last_vrng[id] = {
        'area':  (30000, 50000),
        'ratio': (0.03, 0.05),
      }

      self.var_status.set(self._mem_info_str('Load'))

    kind = self.var_kind.get()
    vmin, vmax = self.cache[id]['vrng'][kind]
    res = RESOLUTION[kind]
    vmin = math.floor(vmin / res) * res
    vmax = math.ceil (vmax / res) * res
    self.sc_min.config(from_=vmin, to=vmax, resolution=res)
    self.sc_max.config(from_=vmin, to=vmax, resolution=res)
    vmin, vmax = self.last_vrng[id][kind]
    self.var_vmin.set(vmin)
    self.var_vmax.set(vmax)

    self.redraw()

  def redraw(self):
    try: id = self.ls.selection_get()
    except: return
    if id not in self.cache: return
    cfg = self.cache[id]
    vmax = self.var_vmax.get()
    vmin = self.var_vmin.get()
    if vmax < vmin: return
    kind = self.var_kind.get()

    annots = [annot for annot in cfg['annotations'] if vmin <= annot[kind] <= vmax]
    new_annots_ids = {annot['id'] for annot in annots}
    if self.cur_annots_ids == new_annots_ids: return
    else: self.cur_annots_ids = new_annots_ids

    for annot in annots:  # lazy decode
      if isinstance(annot['mask'], ndarray): continue
      annot['mask'] = np.ascontiguousarray(decode(annot['mask']), dtype=bool)  # NOTE: must be bool
    sorted_annots = sorted(annots, key=(lambda x: x[kind]), reverse=True)   # small lay-over big 
    self.var_info.set(f'>> mask filtered {len(annots)} / {len(cfg["annotations"])}')

    img = np.zeros(shape=(*cfg['size'], 4), dtype=np.uint8)
    for annot in sorted_annots:
      img[annot['mask']] = annot['color']

    self.ax.cla()
    self.ax.imshow(img)
    self.cvs.draw()

  def _mem_info_str(self, title='Mem'):
    mem = self.p.memory_info()
    return f'[{title}] rss: {mem.rss//2**20:.3f} MB, vms: {mem.vms//2**20:.3f} MB'

  def save_fig(self):
    try: id = self.ls.selection_get()
    except: return
    if id not in self.cache: return
    vmax = self.var_vmax.get()
    vmin = self.var_vmin.get()
    kind = self.var_kind.get()

    if kind == 'area':
      vmin = int(vmin)
      vmax = int(vmax)

    fn = f'{id}_{kind}_{vmin}~{vmax}.png'
    fp = tkfdlg.asksaveasfilename(initialdir=str(OUT_PATH), initialfile=fn, defaultextension='.png', filetypes=[('PNG Files', '.png')])
    if not fp: return
    fp = Path(fp)
    if not fp.suffix: fp = fp.with_suffix('.png')
    self.fig.savefig(fp, dpi=600)

  def mem_stats(self):
    self.var_status.set(self._mem_info_str())

  def mem_clean(self):
    self.ls.selection_clear(0, tk.END)

    self.cache.clear()
    gc.collect()
    self.var_status.set(self._mem_info_str('Clean'))


if __name__ == '__main__':
  App()
