import pickle

import cv2
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.ndimage import median_filter
from scipy.signal import medfilt2d
from FUSION.interface.interface_tools import *
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

from FUSION.tools.mapping_tools import map_distance, orientation_calibration
from FUSION.tools.method_fusion import *
from FUSION.tools.registration_tools import *
from FUSION.interface.Control_panel import ControlPanel
from FUSION.tools.data_management_tools import register_cmap_Lynred, register_cmap


class BaseFramework(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.master.protocol("WM_DELETE_WINDOW", self._quit_app)
        self.place(x=0, y=0)
        self._specialized(master)
        self.master.attributes('-fullscreen', False)
        self.fullScreenState = False
        self.master.bind("<F11>", self._toggleFullScreen)
        self.master.bind("<Escape>", self._quitFullScreen)

    def _toggleFullScreen(self, event=None):
        self.fullScreenState = not self.fullScreenState
        self.master.attributes("-fullscreen", self.fullScreenState)

    def _quitFullScreen(self, event=None):
        self.fullScreenState = False
        self.master.attributes("-fullscreen", self.fullScreenState)

    def _specialized(self, master):
        ##
        # Define the needed Canvas / Control Panel and Menu
        ##
        self.master.title("Basic Window app")
        # Canvas
        # If needed
        # Menu
        self._menu(master)

    def _menu(self, master):
        ##
        # Creation of the Menu
        #
        menubar = tk.Menu(master)
        File = tk.Menu(menubar, tearoff=0)
        File.add_separator()
        File.add_command(label="Open a Door", command=self._alert)
        File.add_command(label="Quit", command=self._quit_app)
        menubar.add_cascade(label="File", menu=File)

    def _quit_app(self):
        ##
        # Application quit function
        #
        Msgbox = tk.messagebox.askquestion("Exit Application", "Are you sure?", icon="warning")
        if Msgbox == "yes":
            self.quit()
            self.master.destroy()
        else:
            pass

    def _alert(self):
        print("The Door is opened")


class Application(BaseFramework):
    def __init__(self, master):
        self._calibration = tk.BooleanVar()
        super().__init__(master)
        self._activate(None)
        self._toggleFullScreen(None)
        self.win = None
        self.IR_pipe = {}
        self.VIS_pipe = {}
        self._orientation_loading()

    # Creation of the widgets #################################################################################
    def _menu(self, master):
        menubar = tk.Menu(master)
        File = tk.Menu(menubar, tearoff=0)
        File.add_command(label="Open...", command=self._open_image)
        File.add_command(label="Open random...", command=self._open_image_random)
        File.add_command(label="Open manually...", command=self._open_image_manually)
        File.add_command(label="Save...", command=self._alert)
        File.add_separator()
        File.add_command(label="Quit", command=self._quit_app)
        menubar.add_cascade(label="File", menu=File)

        Fusion = tk.Menu(menubar, tearoff=0)
        Fusion.add_command(label="ColorMap fusion", command=self._colormap_fusion_panel)
        Fusion.add_command(label="simple Luminance fusion", command=self._luminance_fusion_panel)
        Fusion.add_command(label="Gradient map fusion", command=self._gradient_map_fusion_panel)
        Fusion.add_command(label="Multi-scale fusion", command=self._multiscale_fusion_panel)
        menubar.add_cascade(label="Fusion", menu=Fusion)

        Calibration = tk.Menu(menubar, tearoff=0)
        Calibration.add_command(label='Orientation Estimation', command=lambda: self._orientation_calibration(one=True))
        Calibration.add_command(label='Orientation Estimation (10 images)', command=self._orientation_calibration)
        Calibration.add_command(label='Manual Calibration', command=self._manual_calibration)
        Calibration.add_separator()
        Calibration.add_command(label='Toggle Calibration', command=self._toggleCalibration)
        menubar.add_cascade(label="Calibration", menu=Calibration)

        Display = tk.Menu(menubar, tearoff=0)
        Display.add_command(label="Toggle Fullscreen", command=self._toggleFullScreen)
        Display.add_command(label='Reset Selection to original', command=self._reset_selection)
        menubar.add_cascade(label="Display", menu=Display)

        self._select = tk.IntVar(self)
        self._select.set(3)
        self._select_change = tk.BooleanVar(self)
        self._select_change.set(False)

        Selection = tk.Menu(menubar, tearoff=0)
        Selection.add_radiobutton(label="Visible", value=2, variable=self._select, command=self._selectChange)
        Selection.add_radiobutton(label="IR", value=1, variable=self._select, command=self._selectChange)
        Selection.add_radiobutton(label="Both", value=3, variable=self._select, command=self._selectChange)
        menubar.add_cascade(label="Selection", menu=Selection)
        self.image_process = tk.BooleanVar()
        self.image_process.set(0)

        menubar.add_command(label='Image Processing Tool...', command=self._image_processing_tools)
        menubar.add_command(label='Depth Mapping...', command=self._depth_map_Window)

        master.configure(menu=menubar)

    def _opening_screen(self, master):
        self.master.geometry("1920x1080")
        # Set up the Horizontal paned Window
        self._images_frame = tk.Frame(master, bg='white', width=660, height=1080)
        self._images_frame.grid_propagate(0)
        self._fusion_frame = tk.Frame(master, bg='white', width=1320, height=1080)
        self._images_frame.place(x=0, y=0)
        self._fusion_frame.place(x=660, y=0)

        # Set up the Frame for visible and infrared
        self._frame_ir = tk.LabelFrame(self._images_frame, text='Infrared', bg='white', fg='black',
                                       highlightcolor='red',
                                       width=652, height=535)
        self._frame_ir.pack()
        self._frame_vis = tk.LabelFrame(self._images_frame, text='Visible', bg='white', fg='black',
                                        highlightcolor='red',
                                        width=652, height=535)
        self._frame_vis.pack()
        self._frame_fus = tk.LabelFrame(self._fusion_frame, text='Fusion', bg='white', fg='black', width=1250,
                                        height=900)
        self._frame_fus.pack()
        self._canvas_IR = tk.Canvas(self._frame_ir, width=652, height=492, background='black')
        self._canvas_VIS = tk.Canvas(self._frame_vis, width=652, height=492, background='white')
        self._canvas_FUS = tk.Canvas(self._frame_fus, width=1240, height=800, background='white')
        self._canvas_IR.pack()
        self._canvas_VIS.pack()
        self._canvas_FUS.pack()

        vis, ir, fus = random_image_opening()
        self.VIS = ImageCustom(vis)
        self.IR = ImageCustom(ir)
        self.IR_origin, self.VIS_origin = size_matcher(self.IR, self.VIS)
        self.IR_current_value, self.VIS_current_value = self.IR_origin.copy(), self.VIS_origin.copy()
        self.FUS = ImageCustom(fus)
        self._calibrate()
        self._vis = prepare_image(self.VIS)
        self._fus = prepare_image(self.FUS, size=(1280, 960))
        self._canvas_VIS.create_image(328, 248, image=self._vis)
        self._canvas_FUS.create_image(650, 490, image=self._fus)

    def _orientation_loading(self):
        p = join(abspath(dirname('tools')), 'tools', 'calibration_selection')
        try:
            if os.stat(join(p, 'angle_calibration')).st_size > 0:
                with open(join(p, 'angle_calibration'), "rb") as f:
                    self.angle = pickle.load(f)
                with open(join(p, 'length_calibration'), "rb") as f:
                    self.gap = pickle.load(f)
            else:
                self.angle = 0
                self.gap = (0, 0)
        except OSError:
            print("There is no Calibration file here !")
            self.angle = 0
            self.gap = (0, 0)

    def _toggleCalibration(self):
        if self._calibration.get():
            self._calibration.set(0)
            self.IR_current_value = self._apply_pipe(self.IR_origin, 'ir')
            self._refresh(self.IR_current_value, 'ir')
        else:
            self._calibration.set(1)
            self.IR_current_value = self._apply_pipe(self.IR_origin, 'ir')
            self._calibrate()
            self._refresh(self.IR_current_value, 'ir')
        if self._current_panel == 'colormap_fusion':
            self._colormap_change()
        elif self._current_panel == 'gradient_map_fusion':
            self._filter_panel()
        elif self._current_panel == 'luminance_fusion':
            self._domain_change()
        elif self._current_panel == 'multiscale_fusion':
            self._fusion()

    def _specialized(self, master):
        self.master.title("Image Fusion App")
        # Canvas
        self._opening_screen(master)
        # Menu
        self._menu(master)

    # #########################################################################################################
    # Selection of the active images ##########################################################################
    def _selectChange(self):
        self._select_change.set(not self._select_change.get())
        self._unlight()
        if self._select.get() == 1 or self._select.get() == 3:
            highlight(self._canvas_IR)
        if self._select.get() == 2 or self._select.get() == 3:
            highlight(self._canvas_VIS)

    # #########################################################################################################

    # Image management tools ##################################################################################
    def _open_image(self):
        p = dirname(abspath('FUSION'))
        filename = askopenfilename(title="Open an image", filetypes=[('jpg files', '.jpg'), ('png files', '.png'),
                                                                     ('all files', '.*')],
                                   initialdir=join(p, 'Images_grouped', 'visible'))
        num = search_number(filename)
        ext = search_ext(join(p, 'Images_grouped', 'infrared'), num)
        filename_ir = p + "/Images_grouped/infrared/IFR_" + num + ext
        self.VIS = ImageCustom(filename)
        self._vis = prepare_image(self.VIS)
        self.IR = ImageCustom(filename_ir)
        self.IR_origin, self.VIS_origin = size_matcher(self.IR, self.VIS)
        self.IR_current_value, self.VIS_current_value = self.IR_origin.copy(), self.VIS_origin.copy()
        self._calibrate()
        self._canvas_VIS.create_image(328, 248, image=self._vis)

    def _open_image_random(self):
        vis, ir, fus = random_image_opening()
        self.VIS = ImageCustom(vis)
        self._vis = prepare_image(self.VIS)
        self.IR = ImageCustom(ir)
        self.IR_origin, self.VIS_origin = size_matcher(self.IR, self.VIS)
        self.IR_current_value, self.VIS_current_value = self.IR_origin.copy(), self.VIS_origin.copy()
        self._calibrate()
        self._canvas_VIS.create_image(328, 248, image=self._vis)

    def _open_image_manually(self):
        p = dirname(abspath('FUSION'))
        filename_vis = askopenfilename(title="Open an image Visible",
                                       filetypes=[('jpg files', '.jpg'), ('png files', '.png'),
                                                  ('all files', '.*')],
                                       initialdir=join(p, 'Images_grouped', 'visible'))
        filename_ir = askopenfilename(title="Open an image Infrared",
                                      filetypes=[('tiff files', '.tiff'), ('png files', '.png'),
                                                 ('all files', '.*')],
                                      initialdir=join(p, 'Images_grouped', 'infrared'))
        self.VIS = ImageCustom(filename_vis)
        self._vis = prepare_image(self.VIS)
        self.IR = ImageCustom(filename_ir)
        self.IR_origin, self.VIS_origin = size_matcher(self.IR, self.VIS)
        self.IR_current_value, self.VIS_current_value = self.IR_origin.copy(), self.VIS_origin.copy()
        self._calibrate()
        self._canvas_VIS.create_image(328, 248, image=self._vis)

    # #########################################################################################################
    # Fusion Panel Definition #################################################################################
    def _activate(self, name_of_panel):
        if name_of_panel:
            self._current_panel = name_of_panel
        else:
            self._current_panel = None

    def _colormap_fusion_panel(self):
        global cmaps
        if self._current_panel == 'colormap_fusion':
            pass
        else:
            if hasattr(self, "_controlPanel"):
                self._controlPanel.destroy()
            self._canvas_FUS.destroy()
            self._canvas_FUS = tk.Canvas(self._frame_fus, width=1250, height=900, background='white')
            self._canvas_FUS.pack()
            self._controlPanel = tk.Frame(self._frame_fus, bg='white', height=150, width=1250)
            self._controlPanel.pack()
            register_cmap_Lynred('D:\Travail\LYNRED\FUSION\interface\data_interface\LUT8_lifeinred_color.dat')
            p = join(dirname(abspath('data_interface')), "interface", "data_interface")
            viridis, plasma, inferno, magma, cividis, life_inred = io.imread(p + "/viridis.png"), io.imread(
                p + "/plasma.png"), \
                                                                   io.imread(p + "/inferno.png"), io.imread(
                p + "/magma.png"), \
                                                                   io.imread(p + "/cividis.png"), io.imread(
                p + "/life_inred.png")
            cmaps = {'viridis': viridis,
                     'plasma': plasma,
                     'inferno': inferno,
                     'magma': magma,
                     'cividis': cividis,
                     'lifeinred_color': life_inred}
            self._list_colormap = ttk.Combobox(self._controlPanel, values=list(cmaps.keys()), state="readonly")
            self._list_colormap.current(2)
            self._list_colormap.bind("<<ComboboxSelected>>", self._colormap_change)
            self._list_colormap.place(x=5, y=0)
            self._colormap_selected = tk.Canvas(self._controlPanel, width=500, height=20, background='white')
            self._colormap_selected.place(x=155, y=0)
            self._slide_ratio = tk.Scale(self._controlPanel, from_=0, to=100, orient=tk.HORIZONTAL, length=400,
                                         command=self._fusion)
            self._slide_ratio.place(x=5, y=50)
            self._slide_ratio.set(50)
            self._label_ratio = tk.Label(self._controlPanel, anchor='nw',
                                         text='Visible', background='white')
            self._label_ratio.place(x=10, y=100)
            self._label_ratio = tk.Label(self._controlPanel, anchor='nw',
                                         text='Infrared', background='white')
            self._label_ratio.place(x=380, y=100)
            self._colorbar_selected = prepare_image(cmaps[self._list_colormap.get()], size=(500, 20))
            self._colormap_selected.create_image(250, 10, image=self._colorbar_selected)
            self._activate("colormap_fusion")
            self._colormap_change("<<ComboboxSelected>>")

    def _luminance_fusion_panel(self):
        global color_domain
        if self._current_panel == 'luminance_fusion':
            pass
        else:
            if hasattr(self, "_controlPanel"):
                self._controlPanel.destroy()
            self._canvas_FUS.destroy()
            self._canvas_FUS = tk.Canvas(self._frame_fus, width=1250, height=900, background='white')
            self._canvas_FUS.pack()
            self._controlPanel = tk.Frame(self._frame_fus, bg='white', height=150, width=1250)
            self._controlPanel.pack()

            color_domain = ['HSV', 'LAB']
            self._list_color_domain = ttk.Combobox(self._controlPanel, values=color_domain, state="readonly")
            self._list_color_domain.current(1)
            self._list_color_domain.bind("<<ComboboxSelected>>", self._domain_change)
            self._list_color_domain.place(x=5, y=0)

            self._slide_ratio = tk.Scale(self._controlPanel, from_=0, to=100, orient=tk.HORIZONTAL, length=400,
                                         command=self._fusion)
            self._slide_ratio.place(x=5, y=50)
            self._slide_ratio.set(50)
            self._label_ratio = tk.Label(self._controlPanel, anchor='nw',
                                         text='Visible', background='white')
            self._label_ratio.place(x=10, y=100)
            self._label_ratio = tk.Label(self._controlPanel, anchor='nw',
                                         text='Infrared', background='white')
            self._label_ratio.place(x=380, y=100)
            self._template_button = tk.Button(self._controlPanel, text='Template matching', command=self._depth_map)
            self._template_button.place(x=800, y=100)

            self._activate("luminance_fusion")
            self._domain_change("<<ComboboxSelected>>")

    def _gradient_map_fusion_panel(self):
        global filter_gradient
        if self._current_panel == 'gradient_map_fusion':
            pass
        else:
            if hasattr(self, "_controlPanel"):
                self._controlPanel.destroy()
            self._canvas_FUS.destroy()
            self._canvas_FUS = tk.Canvas(self._frame_fus, width=1250, height=900, background='white')
            self._canvas_FUS.pack()
            self._controlPanel = tk.Frame(self._frame_fus, bg='white', height=150, width=1250)
            self._controlPanel.pack()

            ## Cascade list of filter
            filter_gradient = ['Prewitt', 'Sobel', 'Canny', 'Laplacian', 'Roberts', 'Perso']
            self._list_filter_gradient = ttk.Combobox(self._controlPanel, values=filter_gradient, state="readonly")
            self._list_filter_gradient.current(0)
            self._list_filter_gradient.bind("<<ComboboxSelected>>", self._filter_panel)
            self._list_filter_gradient.place(x=5, y=0)

            ## Choose of the output image
            global IR_edges
            global Full_edges
            IR_edges = tk.BooleanVar(self)
            IR_edges.set(0)
            Full_edges = tk.BooleanVar(self)
            Full_edges.set(0)
            self._IR_edges_check = tk.Checkbutton(self._controlPanel, variable=IR_edges, text='Project on Infrared',
                                                  command=self._fusion, bg='white')
            self._IR_edges_check.place(x=950, y=25)
            self._Full_edges_check = tk.Checkbutton(self._controlPanel, variable=Full_edges,
                                                    text='Display only the edges',
                                                    command=self._fusion, bg='white')
            self._Full_edges_check.place(x=950, y=55)
            self._template_button = tk.Button(self._controlPanel, text='Depth Map', command=self._depth_map)
            self._template_button.place(x=800, y=100)
            self._select.set(3)
            self._selectChange()
            self._activate("gradient_map_fusion")
            # place widgets
            self._filter_panel()
        # method="Canny", kernel_size=5, kernel_blur=3, low_threshold=5, high_threshold=15

    def _multiscale_fusion_panel(self):
        if self._current_panel == 'multiscale_fusion':
            pass
        else:
            if hasattr(self, "_controlPanel"):
                self._controlPanel.destroy()
            self._canvas_FUS.destroy()
            self._canvas_FUS = tk.Canvas(self._frame_fus, width=1250, height=900, background='white')
            self._canvas_FUS.pack()
            self._controlPanel = tk.Frame(self._frame_fus, bg='white', height=150, width=1250)
            self._controlPanel.pack()
            p = join(dirname(abspath('data_interface')), "interface", "data_interface")
            scale = ['Harr', 'Wavelet']
            self._list_scale_method = ttk.Combobox(self._controlPanel, values=scale, state="readonly")
            self._list_scale_method.current(0)
            self._list_scale_method.bind("<<ComboboxSelected>>", self._domain_change)
            self._list_scale_method.place(x=5, y=0)
            self._slide_ratio = tk.Scale(self._controlPanel, from_=0, to=100, orient=tk.HORIZONTAL, length=400,
                                         command=self._fusion)
            self._slide_ratio.place(x=5, y=50)
            self._slide_ratio.set(50)
            self._label_ratio = tk.Label(self._controlPanel, anchor='nw',
                                         text='Visible', background='white')
            self._label_ratio.place(x=10, y=100)
            self._label_ratio = tk.Label(self._controlPanel, anchor='nw',
                                         text='Infrared', background='white')
            self._label_ratio.place(x=380, y=100)
            self._level_ratio = tk.Scale(self._controlPanel, from_=0, to=5, orient=tk.HORIZONTAL, length=400,
                                         command=self._fusion)
            self._level_ratio.place(x=455, y=50)
            self._level_ratio.set(2)
            self._activate("multiscale_fusion")
            self._domain_change("<<ComboboxSelected>>")

    def _colormap_change(self, event=None):
        if self._select.get() == 1 or self._select.get() == 2:
            self._select.set(3)
            self._selectChange()
        self._fusion()
        self._colorbar_selected = prepare_image(cmaps[self._list_colormap.get()], size=(500, 20))
        self._colormap_selected.create_image(250, 10, image=self._colorbar_selected)

    def _domain_change(self, event=None):
        if self._select.get() == 1 or self._select.get() == 2:
            self._select.set(3)
            self._selectChange()
        self.IR, self.VIS = self.IR_current_value, self.VIS_current_value
        self._refresh(self.IR, 'ir')
        self._refresh(self.VIS, 'vis')
        self._fusion()

    def _filter_panel(self, event=None):
        if self._select.get() != 3:
            self._select.set(3)
            self._selectChange()
        for widget in self._controlPanel.winfo_children():
            if widget.winfo_class() == 'Scale' or widget.winfo_class() == 'Label':
                widget.destroy()
        # Scale for the ratio VIS / IR
        self._slide_ratio = tk.Scale(self._controlPanel, from_=0, to=100, orient=tk.HORIZONTAL, length=400,
                                     command=self._fusion, bg='white')
        self._slide_ratio.place(x=5, y=70)
        self._slide_ratio.set(50)
        self._label_ratio1 = tk.Label(self._controlPanel, anchor='nw',
                                      text='Visible', background='white')
        self._label_ratio1.place(x=10, y=120)
        self._label_ratio2 = tk.Label(self._controlPanel, anchor='nw',
                                      text='Infrared', background='white')
        self._label_ratio2.place(x=380, y=120)
        # Scale for the kernel Blur size
        self._kernel_blur = tk.Scale(self._controlPanel, from_=0, to=16, orient=tk.HORIZONTAL, length=100,
                                     command=self._fusion, bg='white')
        self._kernel_blur.place(x=450, y=25)
        self._kernel_blur.set(1)
        self._label_kblur = tk.Label(self._controlPanel, anchor='nw', text='Kernel Blur radius', background='white')
        self._label_kblur.place(x=445, y=70)

        if self._list_filter_gradient.get() == 'Perso':
            self._level = tk.Scale(self._controlPanel, from_=0, to=10, orient=tk.HORIZONTAL, length=150,
                                   command=self._fusion, bg='white')
            self._level.place(x=600, y=25)
            self._level.set(3)
            self._label_level = tk.Label(self._controlPanel, anchor='nw',
                                         text='Level of the Pyramid', background='white')
            self._label_level.place(x=630, y=70)

        if self._list_filter_gradient.get() == 'Canny':
            self._th_low = tk.Scale(self._controlPanel, from_=0, to=40, orient=tk.HORIZONTAL, length=150,
                                    command=self._fusion, bg='white')
            self._th_low.place(x=600, y=25)
            self._th_low.set(5)
            self._label_th = tk.Label(self._controlPanel, anchor='nw',
                                      text='Low Threshold', background='white')
            self._label_th.place(x=630, y=70)
            self._th_ratio = tk.Scale(self._controlPanel, from_=0, to=15, orient=tk.HORIZONTAL, length=100,
                                      command=self._fusion, bg='white')
            self._th_ratio.place(x=770, y=25)
            self._th_ratio.set(3)
            self._label_th_ratio = tk.Label(self._controlPanel, anchor='nw',
                                            text='Ratio High/Low Threshold', background='white')
            self._label_th_ratio.place(x=760, y=70)

    def _fusion(self, event=None):
        # print(self.IR_current_value)
        if self._select_change.get():
            self._selectChange()
            if self._current_panel == 'colormap_fusion':
                self._colormap_change("<<ComboboxSelected>>")
            elif self._current_panel == 'luminance_fusion':
                self._domain_change("<<ComboboxSelected>>")
            elif self._current_panel == 'gradient_map_fusion':
                self._filter_panel()
            elif self._current_panel == 'multiscale_fusion':
                self._multiscale_fusion_panel()
        else:
            ratio = self._slide_ratio.get() / 100
            if self._current_panel == 'colormap_fusion':
                self._unlight()
                self.FUS, self.IR = colormap_fusion(self.IR_current_value, self.VIS_current_value,
                                                    ratio=self._slide_ratio.get() / 100,
                                                    colormap=self._list_colormap.get())
                self._refresh(self.IR, 'ir')
                self._refresh()
            elif self._current_panel == 'luminance_fusion':
                self._unlight()
                domain = self._list_color_domain.get()
                self.FUS = grayscale_fusion(self.IR_current_value, self.VIS_current_value, ratio=ratio, domain=domain)
            elif self._current_panel == "gradient_map_fusion":
                if self._list_filter_gradient.get() == 'Laplacian' and self._list_filter_gradient.get() == 'Sobel':
                    self.IR, self.VIS = region_based_fusion(self.IR_current_value, self.VIS_current_value,
                                                            method=self._list_filter_gradient.get(),
                                                            kernel_blur=self._kernel_blur.get() * 2 + 1)
                elif self._list_filter_gradient.get() == 'Canny':
                    self.IR, self.VIS = region_based_fusion(self.IR_current_value, self.VIS_current_value,
                                                            method='Canny',
                                                            kernel_blur=self._kernel_blur.get() * 2 + 1,
                                                            low_threshold=self._th_low.get(),
                                                            ratio=self._th_ratio.get() / 4)
                elif self._list_filter_gradient.get() == 'Perso':
                    self.IR, self.VIS = region_based_fusion(self.IR_current_value, self.VIS_current_value,
                                                            method='Perso',
                                                            kernel_blur=self._kernel_blur.get() * 2 + 1,
                                                            level=self._level.get())
                else:
                    self.IR, self.VIS = region_based_fusion(self.IR_current_value, self.VIS_current_value,
                                                            method=self._list_filter_gradient.get(),
                                                            kernel_blur=self._kernel_blur.get() * 2 + 1)
                self.FUS = self.VIS_current_value.LAB()
                if IR_edges.get() and not Full_edges.get():
                    temp = fusion_scaled(self.IR, fusion_scaled(self.IR, self.VIS, ratio=ratio), ratio=1) + \
                           fusion_scaled(self.IR, fusion_scaled(self.IR, self.VIS, ratio=ratio), ratio=0)
                else:
                    temp = np.float_(self.FUS[:, :, 0]) + fusion_scaled(self.IR, self.VIS, ratio=ratio)
                temp[temp > 255] = 255
                temp[temp < 0] = 0
                self.FUS[:, :, 0] = np.uint8(temp)
                self.FUS = self.FUS.RGB()
                if Full_edges.get():
                    temp1 = np.uint8(128 + np.float_(self.IR) / 2)
                    temp2 = np.uint8(128 - np.float_(self.VIS) / 2)
                    self.IR, self.VIS = colormap_fusion(temp1, temp2, ratio=-1, colormap='twilight')
                    self.FUS = ImageCustom(fusion_scaled(self.IR, self.VIS, ratio=ratio))
                self._refresh(self.IR, 'ir')
                self._refresh(self.VIS, 'vis')
            elif self._current_panel == 'multiscale_fusion':
                self._unlight()
                scale_method = self._list_scale_method.get()
                if scale_method == 'Harr':
                    self.FUS = Harr_fus(self.IR_current_value, self.VIS_current_value,
                                        ratio=ratio, level=self._level_ratio.get())
                else:
                    pass
                    # self.FUS = Wavelet_fus(self.IR_current_value, self.VIS_current_value, ratio=ratio, domain=level)
            self._refresh(size=(1280, 960))



    # ########################################################################################################

    # Calibration tools ######################################################################################
    def _calibrate(self):
        p = join(abspath(dirname('tools')), 'tools', 'calibration_selection')
        try:
            if os.stat(join(p, 'transform_matrix')).st_size > 0:
                with open(join(p, 'transform_matrix'), "rb") as f:
                    tform = pickle.load(f)
            else:
                tform = np.zeros([3, 3])
        except OSError:
            print("There is no Transformation matrix there !")
            tform = np.zeros([3, 3])
        if tform.sum() == 0:
            self.IR, self.VIS = size_matcher(self.IR, self.VIS)
        else:
            _, self.VIS = size_matcher(self.IR, self.VIS)
            height, width = self.VIS.shape[:2]
            temp = cv.warpPerspective(self.IR_current_value, tform, (width, height))
            self.IR_current_value = ImageCustom(cv.pyrDown(temp), self.IR_origin)
            self.IR = ImageCustom(cv.pyrDown(temp), self.IR)
        self._calibration.set(1)
        self._ir = prepare_image(self.IR)
        clearImage(self._canvas_FUS)
        self._canvas_IR.create_image(328, 248, image=self._ir)

    def _manual_calibration(self):
        # selection = [32, 55]  # , 204, 230, 573]
        # p = join(abspath(dirname('tools')), 'tools', 'calibration_selection')
        pts_src = []
        pts_dst = []
        tform = np.zeros([3, 3])
        # for num in selection:
        image_ir = self.IR_origin
        image_vis = self.VIS_origin
        # image_ir = ImageCustom(join(p, 'IFR_' + str(num) + '.tiff'))
        # image_vis = ImageCustom(join(p, 'VIS_' + str(num) + '.jpg'))
        # image_ir, image_vis = size_matcher(image_ir, image_vis)
        if len(pts_src) == 0:
            pts_src, pts_dst, tform_temp = manual_calibration(image_ir, image_vis)
        else:
            pts_src_temp, pts_dst_temp, tform_temp = manual_calibration(image_ir, image_vis)
            np.append(pts_src, pts_src_temp, axis=0)
            np.append(pts_dst, pts_dst_temp, axis=0)
        tform += tform_temp
        # tform /= len(selection)
        tform_b, status = cv.findHomography(pts_src_temp, pts_dst_temp)
        height, width = image_vis.shape[:2]
        choice1 = ImageCustom(cv.warpPerspective(image_ir, tform, (width, height)), image_ir)
        choice2 = ImageCustom(cv.warpPerspective(image_ir, tform_b, (width, height)), image_ir)
        choice1 = choice1 / 2 + image_vis.GRAYSCALE() / 2
        choice2 = choice2 / 2 + image_vis.GRAYSCALE() / 2
        I1 = image_vis.LAB().copy()
        I2 = I1.copy()
        I1[:, :, 0] = choice1
        I2[:, :, 0] = choice2
        choice1 = cv.cvtColor(I1, cv.COLOR_LAB2BGR)
        choice2 = cv.cvtColor(I2, cv.COLOR_LAB2BGR)
        choice = choose(choice1, choice2)
        with open(join(p, "transform_matrix"), "wb") as p:
            if choice == 1:
                pickle.dump(tform, p)
            else:
                pickle.dump(tform_b, p)
        self._calibrate()

    def _refresh(self, *args, size=(1280, 960)):
        if len(args) == 0 or len(args) == 1 or len(args) > 2:
            self._fus = prepare_image(self.FUS, size)
            self._canvas_FUS.create_image(size[0] / 2, size[1] / 2, image=self._fus)
        elif not isinstance(args[0], (ImageCustom, np.ndarray)):
            raise TypeError("The first argument need to be an instance of ImageCustom")
        elif not (args[1] == 'fus' or args[1] == 'ir' or args[1] == 'vis'):
            raise ValueError("The names to use are 'fus', 'ir', 'vis'")
        else:
            if args[1] == 'fus':
                self._fus = prepare_image(args[0], size)
                self._canvas_FUS.create_image(size[0] / 2, size[1] / 2, image=self._fus)
            elif args[1] == 'ir':
                self._ir = prepare_image(args[0])
                self._canvas_IR.create_image(328, 248, image=self._ir)
            elif args[1] == 'vis':
                self._vis = prepare_image(args[0])
                self._canvas_VIS.create_image(328, 248, image=self._vis)
            else:
                self._fus = prepare_image(self.FUS, size)
                self._canvas_FUS.create_image(size[0] / 2, size[1] / 2, image=self._fus)

    def _depth_map(self):
        self.depth_map = map_distance(self.gap[0], self.gap[1], self.angle, self.IR_origin, self.VIS_origin,
                                      level=4, method='Prewitt', blur_filter=3, threshold=200)

    def _orientation_calibration(self, one=False):
        if one:
            self.angle, self.gap, self.center, _ = orientation_calibration(method='prewitt', L=100,
                                                                           one=one, image_IR=self.IR,
                                                                           image_RGB=self.VIS, verbose=True)
        else:
            self.angle, self.gap, self.center, _ = orientation_calibration(method='prewitt', L=100, image_number=10,
                                                                           verbose=True)
            p = join(abspath(dirname('tools')), 'tools', 'calibration_selection')
            with open(join(p, 'angle_calibration'), "wb") as f:
                pickle.dump(self.angle, f)
            with open(join(p, 'length_calibration'), "wb") as f:
                pickle.dump(self.gap, f)

    def _apply_pipe(self, image, pipe):
        if pipe == 'ir':
            if self.IR_pipe:
                return image
            else:
                return image
        if pipe == 'vis':
            if self.VIS_pipe:
                return image
            else:
                return image

    # ########################################################################################################
    # Other tools ############################################################################################
    def _reset_selection(self):
        if self._select.get() == 1 or self._select.get() == 3:
            self.IR = ImageCustom(self.IR_origin)
            self.IR_current_value = self.IR.copy()
            self._refresh(self.IR, 'ir')

        if self._select.get() == 2 or self._select.get() == 3:
            self.VIS = ImageCustom(self.VIS_origin)
            self.VIS_current_value = self.VIS.copy()
            self._refresh(self.VIS, 'vis')

    def _image_processing_tools(self):
        if self.win is None:
            self.win = tk.Tk()
            ImageProcessing = ImageProcessingWindow(self.win, self, self._select.get())
            ImageProcessing.mainloop()
            self.win = None
            self._refresh(self.IR, 'ir')
            self._refresh(self.VIS, 'vis')
        else:
            pass

    def _unlight(self):
        self._canvas_IR['bg'] = 'white'
        self._canvas_IR['relief'] = 'flat'
        self._canvas_VIS['bg'] = 'white'
        self._canvas_VIS['relief'] = 'flat'

    def _depth_map_Window(self):
        if self.win is None:
            self.win = tk.Tk()
            DepthMap = DepthMapWindow(self.win, self)
            DepthMap.mainloop()
            self.win = None
        else:
            pass
    # ########################################################################################################


class DepthMapWindow(BaseFramework):
    def __init__(self, master, main):
        self.main = main
        self.IR = main.IR_origin.copy()
        self.VIS = main.VIS_origin.copy()
        self.IR_origin = main.IR_origin.copy()
        self.VIS_origin = main.VIS_origin.copy()
        self.IR_current_value = main.IR_current_value.copy()
        self.VIS_current_value = main.VIS_current_value.copy()
        self.map = None
        super().__init__(master)
        self._opening_screen()

    def _specialized(self, master):
        self.master.title("Depth Map computing")
        # Menu
        self._menu(master)

    def _menu(self, master):
        self._select_change = tk.BooleanVar(master)
        self._select_change.set(False)
        self.select_processing = tk.IntVar(master)
        menubar = tk.Menu(master)
        File = tk.Menu(menubar, tearoff=0)
        File.add_separator()
        File.add_command(label="Export to main and quit", command=self._export_to_main)
        File.add_command(label="Quit without changes", command=self._quit_app)
        menubar.add_cascade(label="File", menu=File)

        Display = tk.Menu(menubar, tearoff=0)
        Display.add_command(label="Display source images", command=self._display)
        self.disparity_left = tk.BooleanVar(self)
        self.disparity_left.set(1)
        Display.add_radiobutton(label="Display left Disparity map", variable=self.disparity_left,
                                value=True, command=self._apply)
        Display.add_radiobutton(label="Display left Disparity map", variable=self.disparity_left,
                                value=False, command=self._apply)
        menubar.add_cascade(label="Display", menu=Display)

        master.configure(menu=menubar)

    def _display(self, init=False):
        if self._display_screen:
            self._display_screen.destroy()
            self._display_screen = None
            self.master.geometry("1000x480")
        else:
            self.master.geometry("1000x880")
            self._display_screen = tk.Frame(self.master, bg='white', width=1000, height=395, bd=0)
            self._display_screen.place(x=0, y=485)
            self._frame_IR = tk.LabelFrame(self._display_screen, text='Image IR', bg='white', width=496, height=395)
            self._frame_VIS = tk.LabelFrame(self._display_screen, text='Image Visible', bg='white', width=496,
                                            height=395)
            self._frame_IR.place(x=2, y=0)
            self._frame_VIS.place(x=500, y=0)
            self._canvas_IR = tk.Canvas(self._frame_IR, width=496, height=395, background='black')
            self._canvas_VIS = tk.Canvas(self._frame_VIS, width=496, height=395, background='black')
            self._canvas_IR.pack()
            self._canvas_VIS.pack()
        if not init:
            self._apply()

    def _export_to_main(self):
        Msgbox = tk.messagebox.askquestion("Exit Image processing and export Result", "Are you sure?",
                                           icon="warning")
        if Msgbox == "yes":
            if self.map:
                self.main.map = self.map
            self.quit()
            self.master.destroy()
        else:
            tk.messagebox.showinfo("Return", "you will now return to application screen")

    def _opening_screen(self):
        self.master.geometry("1000x480")
        self._frame = tk.Frame(self.master, bg='white', width=640, height=480, bd=0)
        self._controlPanel = tk.LabelFrame(self.master, text='Control Panel', bg='white', width=260, height=480)
        self._frame.place(x=0, y=0)
        self._controlPanel.place(x=640, y=0)
        self._display_screen = None
        self._display(True)
        # Set up the Canva
        self._canvas = tk.Canvas(self._frame, width=640, height=480, background='black')
        self._canvas.pack()
        self._canva_scale = None
        self._level_scale = None
        self.fig, self.ax = None, None
        # Set up the Slide and other choices
        self._kernel_majority = tk.Scale(self._controlPanel, from_=0, to=10, orient=tk.HORIZONTAL, length=115, bg='white',
                                       command=self._apply, label='Majority filter radius')
        self._kernel_majority.place(x=130, y=90)
        self._kernel_majority.set(2)
        self._threshold = tk.Scale(self._controlPanel, from_=0, to=255, orient=tk.HORIZONTAL, length=110, bg='white',
                                   command=self._apply, label='Threshold')
        self._threshold.place(x=5, y=90)
        self._threshold.set(70)
        self._orientation_weight = tk.Scale(self._controlPanel, from_=0, to=30, orient=tk.HORIZONTAL, length=110, bg='white',
                                   command=self._apply, label='Orientation weight')
        self._orientation_weight.place(x=5, y=150)
        self._orientation_weight.set(1)
        ## FilterPanel definition
        self._filter_panel = tk.LabelFrame(self._controlPanel, text='Filter Tuning', bg='white', width=250, height=200)
        self._filter_panel.place(x=5, y=260)
        ## Cascade list of filter
        self._list_filter_gradient = ['Prewitt', 'Sobel', 'Roberts', 'Perso', 'Perso2']
        self._filter_gradient = ttk.Combobox(self._filter_panel, values=self._list_filter_gradient, state="readonly")
        self._filter_gradient.current(0)
        self._filter_gradient.place(x=100, y=5)
        self._filter_gradient.bind("<<ComboboxSelected>>", self._filterPanel(apply=False))
        ####################
        ## Command for the level of depth computed
        self._level_frame = tk.LabelFrame(self._controlPanel, text='Depth of the map', bg='white', width=250, height=80)
        self._level_frame.place(x=5, y=5)
        self._level = tk.Scale(self._level_frame, from_=3, to=11, orient=tk.HORIZONTAL, length=100, bg='white')
        self._level.place(x=5, y=5)
        self._level.set(10)
        self._apply_button = tk.Button(self._level_frame, text='Apply Level',
                                       font=('Arial', 12), bg='white', fg='black', command=self._change_scale)
        self._apply_button.place(x=130, y=5)
        self._filterPanel(False)
        self._change_scale(False)

    def _filterPanel(self, apply=True):
        for widget in self._filter_panel.winfo_children():
            if widget.winfo_class() == 'Scale' or widget.winfo_class() == 'Label' or widget.winfo_class() == 'Checkbutton':
                widget.destroy()
        label_filter = tk.Label(self._filter_panel, anchor='nw', text='Filter :', background='white',
                                font=('Arial', 10))
        label_filter.place(x=15, y=5)
        self._uni = tk.BooleanVar(self)
        self._uni.set(True)
        self._uniform = tk.Checkbutton(self._filter_panel, variable=self._uni, onvalue=True, offvalue=False,
                                              text='Uniform filtering', background='white', command=self._apply)
        self._uniform.place(x=120, y=90)
        self._norm = tk.BooleanVar(self)
        self._norm.set(True)
        self._normalize_grad = tk.Checkbutton(self._filter_panel, variable=self._norm, onvalue=True, offvalue=False,
                                              text='Gradient scaled', background='white', command=self._apply)
        self._normalize_grad.place(x=120, y=60)
        # Scale for the kernel Blur size
        self._filter_blur = tk.Scale(self._filter_panel, from_=0, to=12, orient=tk.HORIZONTAL, length=100,
                                     command=self._apply, bg='white', label='Filter Blur radius')
        self._filter_blur.place(x=5, y=40)
        self._filter_blur.set(3)

        if self._filter_gradient.get() == 'Perso':
            self._level_filter = tk.Scale(self._filter_panel, from_=0, to=10, orient=tk.HORIZONTAL, length=100,
                                          command=self._apply, bg='white', label='Level of Pyramid')
            self._level_filter.place(x=5, y=110)
            self._level_filter.set(0)
            self._norm.set(True)
            self._threshold.set(33)

        if self._filter_gradient.get() == 'Perso2':
            self._level_filter = tk.Scale(self._filter_panel, from_=1, to=5, orient=tk.HORIZONTAL, length=100,
                                          command=self._apply, bg='white', label='Level of Pyramid')
            self._level_filter.place(x=5, y=110)
            self._level_filter.set(1)
            self._norm.set(True)
            self._threshold.set(33)

        if self._filter_gradient.get() == 'Canny':
            self._th_low = tk.Scale(self._filter_panel, from_=0, to=40, orient=tk.HORIZONTAL, length=100,
                                    command=self._apply, bg='white', label='Low Threshold')
            self._th_low.place(x=5, y=110)
            self._th_low.set(15)
            self._th_ratio = tk.Scale(self._filter_panel, from_=0, to=15, orient=tk.HORIZONTAL, length=100,
                                      command=self._apply, bg='white', label='Ratio High/Low')
            self._th_ratio.place(x=130, y=110)
            self._th_ratio.set(10)
        if apply:
            self._apply()

    def _apply(self, event=None):
        level, l_th, ratio = None, None, None
        if self._filter_gradient.get() == 'Canny':
            l_th, ratio = self._th_low.get(), self._th_ratio.get()
        elif self._filter_gradient.get() == 'Perso' or self._filter_gradient.get() =='Perso2':
            level = self._level_filter.get()
        self.map, self.IR, self.VIS = map_distance(self.main.gap[0], self.main.gap[1], self.main.angle,
                                                   self.IR_origin, self.VIS_origin,
                                                   level=self._level.get(),
                                                   method=self._filter_gradient.get(),
                                                   median=self._kernel_majority.get(),
                                                   threshold=self._threshold.get(),
                                                   level_pyr=level,  # For Perso filter
                                                   l_th=l_th, ratio=ratio,  # For Canny
                                                   blur_filter=self._filter_blur.get() * 2 + 1,  # For each filter
                                                   disparity_left=self.disparity_left.get(),
                                                   return_images=True,
                                                   gradient_scaled=self._norm.get(),
                                                   uniform=self._uni.get())
        self._map_image = prepare_image(self.map.RGB(colormap='scale'), master=self)
        self._canvas.create_image(320, 240, image=self._map_image)
        if self._display_screen:
            self._vis = prepare_image(self.VIS, size=(480, 360), master=self)
            self._ir = prepare_image(self.IR, size=(480, 360), master=self)
            self._canvas_VIS.create_image(240, 180, image=self._vis)
            self._canvas_IR.create_image(240, 180, image=self._ir)

    def _change_scale(self, apply=True):
        self._create_cmap()
        dpi = 20
        figh = 470 / dpi
        figw = 50 / dpi
        if self.ax:
            self.ax.clear()
            self.fig.clear()
        self.fig, self.ax = plt.subplots(nrows=1, figsize=(figw, figh), dpi=dpi)
        self.fig.subplots_adjust(top=1, bottom=0,
                                 left=0, right=1)
        gradient = np.linspace(0, 1, 256)
        gradient = np.vstack((gradient, gradient))
        gradient = np.rot90(gradient)
        self.ax.imshow(gradient, aspect='auto', cmap='scale')
        self.ax.set_axis_off()
        if self._level_scale:
            self._level_scale.destroy()
            self._level_scale = tk.LabelFrame(self.master, text='Depth Scale', bg='white', width=100, height=480)
            self._level_scale.place(x=900, y=0)
            self._canva_scale = FigureCanvasTkAgg(self.fig, master=self._level_scale)
            self._canva_scale.draw()
            self._canva_scale.get_tk_widget().pack()
        else:
            self._level_scale = tk.LabelFrame(self.master, text='Depth Scale', bg='white', width=100, height=480)
            self._level_scale.place(x=900, y=0)
            self._canva_scale = FigureCanvasTkAgg(self.fig, master=self._level_scale)
            self._canva_scale.draw()
            self._canva_scale.get_tk_widget().pack()
        if apply:
            self._apply()

    def _create_cmap(self):
        level = self._level.get()
        if level == 11:
            cmap = plt.get_cmap('gnuplot')
            x = np.linspace(0.0, 1.0, 256)
        else:
            cmap = plt.get_cmap('gnuplot')
            x = np.linspace(0.0, 1.0, level)
        cmap_rgb = cm.get_cmap(cmap)(x)[:, :3]
        cmap_rgb[0, :] = [0, 0, 0]
        register_cmap(cmap_rgb, 'scale')
    # ########################################################################################################


class ImageProcessingWindow(BaseFramework):
    def __init__(self, master, main, selection):
        self.main = main
        self.IR = main.IR_origin.copy()
        self.VIS = main.VIS_origin.copy()
        self.IR_origin = self.IR.copy()
        self.VIS_origin = self.VIS.copy()
        self.IR_current_value = self.IR.copy()
        self.VIS_current_value = self.VIS.copy
        super().__init__(master)
        self.select_processing.set(selection)
        self._opening_screen()
        self._reset_selection()

    def _specialized(self, master):
        self.master.title("Image Processing")
        # Menu
        self._menu(master)

    def _menu(self, master):
        self._select_change = tk.BooleanVar(master)
        self._select_change.set(False)
        self.select_processing = tk.IntVar(master)
        menubar = tk.Menu(master)
        File = tk.Menu(menubar, tearoff=0)
        File.add_separator()
        File.add_command(label="Export to main and quit", command=self._export_to_main)
        File.add_command(label="Quit without changes", command=self._quit_app)
        menubar.add_cascade(label="File", menu=File)

        Image_processing = tk.Menu(menubar, tearoff=0)
        Image_processing.add_command(label='Reset Selection to original', command=self._reset_selection)
        Image_processing.add_command(label='Edges enhancement', command=self._edge_enhancements)
        Image_processing.add_command(label='Median filtering', command=self._median_filtering)
        Histo_Equ = tk.Menu(Image_processing, tearoff=0)
        Histo_Equ.add_command(label='Classic Equalization', command=self._histogram_equalize)
        Histo_Equ.add_command(label='CLAHE Equalization', command=self._CLAHE_equalize)
        Image_processing.add_cascade(label='Histogram Equalization', menu=Histo_Equ)
        menubar.add_cascade(label="Image Processing", menu=Image_processing)

        Selection = tk.Menu(menubar, tearoff=0)
        Selection.add_radiobutton(label="Visible", value=2, variable=self.select_processing,
                                  command=self._opening_screen)
        Selection.add_radiobutton(label="IR", value=1, variable=self.select_processing, command=self._opening_screen)
        Selection.add_radiobutton(label="Both", value=3, variable=self.select_processing, command=self._opening_screen)
        menubar.add_cascade(label="Selection", menu=Selection)
        master.configure(menu=menubar)

    def _export_to_main(self):
        Msgbox = tk.messagebox.askquestion("Exit Image processing and export Result", "Are you sure?", icon="warning")
        if Msgbox == "yes":
            self.IR.origin, self.VIS.origin = self.IR.current_value, self.VIS.current_value
            self.main.IR, self.main.VIS = self.IR, self.VIS
            self.quit()
            self.master.destroy()
        else:
            tk.messagebox.showinfo("Return", "you will now return to application screen")

    def _opening_screen(self):
        # for widget in self.winfo_children():
        #     if widget.t
        if self.select_processing.get() == 1:
            self.master.geometry("640x480")
            self._frame_ir = tk.Frame(self.master, bg='white', width=640, height=480)
            # self._frame_ir.grid_propagate(0)
            self._frame_ir.place(x=0, y=0)
        if self.select_processing.get() == 2:
            self.master.geometry("640x480")
            self._frame_vis = tk.Frame(self.master, bg='white', width=640, height=480)
            # self._frame_vis.grid_propagate(0)
            self._frame_vis.place(x=0, y=0)
        if self.select_processing.get() == 3:
            self.master.geometry("1280x480")
            self._frame_ir = tk.Frame(self.master, bg='white', width=640, height=480)
            self._frame_vis = tk.Frame(self.master, bg='white', width=640, height=480)
            # self._frame_ir.grid_propagate(0)
            self._frame_ir.place(x=0, y=0)
            self._frame_vis.place(x=640, y=0)

        # Set up the Canvas for visible and infrared
        if self.select_processing.get() == 1:
            self._canvas_IR = tk.Canvas(self._frame_ir, width=640, height=480, background='black')
            self._canvas_IR.pack()
        if self.select_processing.get() == 2:
            self._canvas_VIS = tk.Canvas(self._frame_vis, width=640, height=480, background='black')
            self._canvas_VIS.pack()
        if self.select_processing.get() == 3:
            self._canvas_IR = tk.Canvas(self._frame_ir, width=640, height=480, background='black')
            self._canvas_VIS = tk.Canvas(self._frame_vis, width=640, height=480, background='white')
            self._canvas_IR.pack()
            self._canvas_VIS.pack()

        if self.select_processing.get() == 1:
            self._ir = prepare_image(self.IR, master=self)
            self._canvas_IR.create_image(320, 240, image=self._ir)
        if self.select_processing.get() == 2:
            self._vis = prepare_image(self.VIS, master=self)
            self._canvas_VIS.create_image(320, 240, image=self._vis)
        if self.select_processing.get() == 3:
            self.master._ir = prepare_image(self.IR, master=self)
            self._canvas_IR.create_image(320, 240, image=self.master._ir)
            self._vis = prepare_image(self.VIS, master=self)
            self._canvas_VIS.create_image(320, 240, image=self._vis)

    def _reset_selection(self):
        if self.select_processing.get() == 3 or self.select_processing.get() == 1:
            self.IR = self.IR_origin.copy()
            self.IR_current_value = self.IR.copy()

        if self.select_processing.get() == 3 or self.select_processing.get() == 2:
            self.VIS = self.VIS_origin.copy()
            self.VIS_current_value = self.VIS.copy()
        self._refresh()

    def _histogram_equalize(self):
        if self.select_processing.get() == 3 or self.select_processing.get() == 1:
            if self.IR.cmap == 'GRAYSCALE':
                self.IR = ImageCustom(cv.equalizeHist(self.IR_current_value), self.IR)
            elif self.IR.cmap == 'HSV':
                self.IR_current_value[:, :, 2] = cv.equalizeHist(self.IR_current_value[:, :, 2])
                self.IR = ImageCustom(self.IR_current_value, self.IR)
            elif self.IR.cmap == 'LAB':
                self.IR_current_value[:, :, 0] = cv.equalizeHist(self.IR_current_value[:, :, 0])
                self.IR = ImageCustom(self.IR_current_value, self.IR)
            elif self.IR.cmap == 'RGB':
                lab = self.IR.LAB()
                lab[:, :, 0] = cv.equalizeHist(lab[:, :, 0])
                lab = ImageCustom(lab, lab)
                self.IR = lab.RGB()
            elif self.IR.cmap == 'EDGES':
                self.IR = ImageCustom(cv.equalizeHist(self.IR_current_value), self.IR)
            self.IR_current_value = np.asarray(self.IR)
        if self.select_processing.get() == 3 or self.select_processing.get() == 2:
            if self.VIS.cmap == 'GRAYSCALE':
                self.VIS = ImageCustom(cv.equalizeHist(self.VIS_current_value), self.VIS)
            elif self.VIS.cmap == 'HSV':
                self.VIS_current_value[:, :, 2] = cv.equalizeHist(self.VIS_current_value[:, :, 2])
                self.VIS = ImageCustom(self.VIS_current_value, self.VIS)
            elif self.VIS.cmap == 'LAB':
                self.VIS_current_value[:, :, 0] = cv.equalizeHist(self.VIS_current_value[:, :, 0])
                self.VIS = ImageCustom(self.VIS_current_value, self.VIS)
            elif self.VIS.cmap == 'RGB':
                lab = self.VIS.LAB()
                lab[:, :, 0] = cv.equalizeHist(lab[:, :, 0])
                lab = ImageCustom(lab, lab)
                self.VIS = lab.RGB()
            elif self.VIS.cmap == 'EDGES':
                self.VIS = ImageCustom(cv.equalizeHist(self.VIS_current_value), self.VIS)
            self.VIS_current_value = np.asarray(self.VIS)
        self._refresh()

    def _CLAHE_equalize(self):
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        if self.select_processing.get() == 3 or self.select_processing.get() == 1:
            if self.IR.cmap == 'GRAYSCALE':
                self.IR = ImageCustom(clahe.apply(self.IR_current_value), self.IR)
            elif self.IR.cmap == 'HSV':
                self.IR_current_value[:, :, 2] = clahe.apply(self.IR_current_value[:, :, 2])
                self.IR = ImageCustom(self.IR_current_value, self.IR)
            elif self.IR.cmap == 'LAB':
                self.IR_current_value[:, :, 0] = clahe.apply(self.IR_current_value[:, :, 0])
                self.IR = ImageCustom(self.IR_current_value, self.IR)
            elif self.IR.cmap == 'RGB':
                lab = self.IR.LAB()
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                lab = ImageCustom(lab, lab)
                self.IR = lab.RGB()
            elif self.IR.cmap == 'EDGES':
                self.IR = ImageCustom(clahe.apply(self.IR_current_value), self.IR)
            self.IR_current_value = np.asarray(self.IR)
        if self.select_processing.get() == 3 or self.select_processing.get() == 2:
            if self.VIS.cmap == 'GRAYSCALE':
                self.VIS = ImageCustom(clahe.apply(self.VIS_current_value), self.VIS)
            elif self.VIS.cmap == 'HSV':
                self.IR_current_value[:, :, 2] = clahe.apply(self.IR_current_value[:, :, 2])
                self.VIS = ImageCustom(self.IR_current_value, self.VIS)
            elif self.VIS.cmap == 'LAB':
                self.IR_current_value[:, :, 0] = clahe.apply(self.IR_current_value[:, :, 0])
                self.VIS = ImageCustom(self.IR_current_value, self.VIS)
            elif self.VIS.cmap == 'RGB':
                lab = self.VIS.LAB()
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                lab = ImageCustom(lab, lab)
                self.VIS = lab.RGB()
            elif self.VIS.cmap == 'EDGES':
                self.VIS = ImageCustom(clahe.apply(self.VIS_current_value), self.VIS)
            self.VIS_current_value = np.asarray(self.VIS)
        self._refresh()

    def _median_filtering(self):
        if self.select_processing.get() == 3 or self.select_processing.get() == 1:
            self.IR = ImageCustom(medfilt2d(self.IR_current_value, kernel_size=3), self.IR)
            self.IR_current_value = np.asarray(self.IR)
        if self.select_processing.get() == 3 or self.select_processing.get() == 2:
            self.VIS = self.VIS.LAB()

            self.VIS[:, :, 0] = medfilt2d(self.VIS[:, :, 0], kernel_size=3)
            self.VIS = self.VIS.RGB()
            self.VIS_current_value = np.asarray(self.VIS)
        self._refresh()

    def _edge_enhancements(self):
        if self.select_processing.get() == 3 or self.select_processing.get() == 1:
            im = np.float_(self.IR_current_value.copy())
            edge = im - 0.7 * abs(cv.Laplacian(im, cv.CV_64F))
            edge[edge < 0] = im[edge < 0]
            self.IR = ImageCustom(edge, self.IR)
            self.IR_current_value = np.asarray(self.IR)
        if self.select_processing.get() == 3 or self.select_processing.get() == 2:
            self.VIS = self.VIS.LAB()
            im = np.float_(self.VIS.copy()[:, :, 0])
            edge = im - 0.7 * abs(cv.Laplacian(im, cv.CV_64F))
            edge[edge < 0] = im[edge < 0]
            self.VIS[:, :, 0] = edge
            self.VIS = ImageCustom(self.VIS.RGB(), self.VIS.RGB())
            self.VIS_current_value = np.asarray(self.VIS)
        self._refresh()

    def _refresh(self):
        if self.select_processing.get() == 3 or self.select_processing.get() == 1:
            self._ir = prepare_image(self.IR_current_value, master=self)
            self._canvas_IR.create_image(320, 240, image=self._ir)
        if self.select_processing.get() == 3 or self.select_processing.get() == 2:
            self._vis = prepare_image(self.VIS_current_value, master=self)
            self._canvas_VIS.create_image(320, 240, image=self._vis)
