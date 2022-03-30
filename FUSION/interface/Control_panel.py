import cv2
from scipy.ndimage import median_filter
from scipy.signal import medfilt2d
from FUSION.interface.interface_tools import *
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from FUSION.tools.method_fusion import *
from FUSION.tools.registration_tools import *


class ControlPanel(tk.Frame):
    def __init__(self, frame, control_type, main):
        super().__init__(frame)
        self.app = main
        self.create_widget(control_type)


    def create_widget(self, control_type):
        if control_type == "colormap":
            p = join(dirname(abspath('data_interface')), "interface", "data_interface")
            viridis, plasma, inferno, magma, cividis = io.imread(p + "/viridis.png"), io.imread(p + "/plasma.png"), \
                                                       io.imread(p + "/inferno.png"), io.imread(p + "/magma.png"), \
                                                       io.imread(p + "/cividis.png")
            cmaps = {'viridis': viridis,
                     'plasma': plasma,
                     'inferno': inferno,
                     'magma': magma,
                     'cividis': cividis}
            self._list_colormap = ttk.Combobox(self, values=list(cmaps.keys()), state="readonly")
            self._list_colormap.current(2)
            self._list_colormap.bind("<<ComboboxSelected>>", self.app._colormap_change)
            self._list_colormap.place(x=5, y=0)
            self._colormap_selected = tk.Canvas(self, width=500, height=20, background='white')
            self._colormap_selected.place(x=155, y=0)
            self._slide_ratio = tk.Scale(self, from_=0, to=100, orient=HORIZONTAL, length=400,
                                         command=self.app._fusion)
            self._slide_ratio.place(x=5, y=50)
            self._slide_ratio.set(50)
            self._label_ratio = tk.Label(self, anchor='nw',
                                         text='Visible', background='white')
            self._label_ratio.place(x=10, y=100)
            self._label_ratio = tk.Label(self, anchor='nw',
                                         text='Infrared', background='white')
            self._label_ratio.place(x=380, y=100)

            self._colorbar_selected = prepare_image(cmaps[self._list_colormap.get()], size=(500, 20))
            self._colormap_selected.create_image(250, 10, image=self._colorbar_selected)
