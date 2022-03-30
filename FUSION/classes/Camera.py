import os
import pickle
import time
from os import listdir
from os.path import *
from tkinter import ttk
import cv2 as cv
from matplotlib import pyplot as plt
from skimage import io
from .Image import ImageCustom
from ..interface.Application import BaseFramework
import tkinter as tk
import threading
import numpy as np
from ..interface.interface_tools import prepare_image, enableChildren, disableChildren
from imutils.video import VideoStream
from ..tools.data_management_tools import register_cmap_Lynred
from ..tools.gradient_tools import edges_extraction
from ..tools.image_processing_tools import histogram_equalization
from ..tools.manipulation_tools import size_matcher, crop_image, manual_calibration
from ..tools.mapping_tools import orientation_calibration_cam
from ..tools.method_fusion import grayscale_fusion, colormap_fusion2
import lynred_py
from tkinter import messagebox
from time import gmtime, strftime


class Camera(BaseFramework):
    def __init__(self, master, source):

        self.source = source
        self._homography_matrix = self._load_homography()
        self.frame = None
        self.thread = None
        self.stopEvent = None
        self.camera_vis = None
        self.camera_ir = None
        self.frame_vis = np.zeros([1, 1])
        self.frame_ir = np.zeros([1, 1])
        self.Windowsize = 0
        super().__init__(master)
        self._calibration = tk.BooleanVar(self)
        self._calibration.set(True)
        self._orientation_loading()
        self._enable_gradient()
        # store the video stream object and output path, then initialize
        # the most recently read frame, thread for reading frames, and
        # the thread stop event

    def _specialized(self, master):
        ##
        # Define the needed Canvas / Control Panel and Menu
        ##
        self.master.title("Video Fusion")
        # Canvas
        self._create_control_panel(master)
        # Menu
        self._menu(master)

    def _quit_app(self):
        if self.stopEvent:
            self.stopEvent.set()
        if self.camera_vis:
            self.camera_vis.stop()
            self.camera_vis = None
        if self.camera_ir:
            self.camera_ir.stop()
            self.camera_ir = None
        super()._quit_app()

    def _orientation_loading(self):
        p = join(abspath(dirname('tools')), '..', 'tools', 'calibration_selection')
        try:
            if os.stat(join(p, 'angle_calibration_cam')).st_size > 0:
                with open(join(p, 'angle_calibration_cam'), "rb") as f:
                    self._angle = pickle.load(f)
                with open(join(p, 'length_calibration_cam'), "rb") as f:
                    self._gap = pickle.load(f)
                with open(join(p, 'center_calibration_cam'), "rb") as f:
                    self._center = pickle.load(f)
            else:
                self._angle = 0
                self._gap = (0, 0)
                self._center = (0, 0)
        except OSError:
            print("There is no Calibration file here !")
            self._angle = 0
            self._gap = (0, 0)
            self._center = (0, 0)

    def _create_control_panel(self, master):
        #### Global frame settings #####
        width = 495
        self.master.geometry("500x980")
        self.master.config(bg='white')
        # Set up the Panel Windows for toolbox
        self._colormapPanel = tk.LabelFrame(master, text='Colormap panel', bg='white', width=width, height=100)
        self._colormapPanel.grid_propagate(0)
        self._colormapPanel.place(x=0, y=5)
        self._domainPanel = tk.LabelFrame(master, text='Color domain panel', bg='white', width=width, height=80)
        self._domainPanel.grid_propagate(0)
        self._domainPanel.place(x=0, y=110)
        self._gradientPanel = tk.LabelFrame(master, text='Gradient panel', bg='white', width=width, height=180)
        self._gradientPanel.grid_propagate(0)
        self._gradientPanel.place(x=0, y=195)
        self._fusionPanel = tk.LabelFrame(master, text='Fusion Panel', bg='white', width=width, height=180)
        self._fusionPanel.grid_propagate(0)
        self._fusionPanel.place(x=0, y=380)

        #################################### COLOR MAP FUSION BOX ##########################################
        register_cmap_Lynred('D:\Travail\LYNRED\FUSION\interface\data_interface\LUT8_lifeinred_color.dat')
        p = join(dirname(abspath('FUSION')), "..", "interface", "data_interface")
        viridis, plasma, inferno, magma, cividis, life_inred = \
            io.imread(p + "/viridis.png"), io.imread(p + "/plasma.png"), \
            io.imread(p + "/inferno.png"), io.imread(p + "/magma.png"), \
            io.imread(p + "/cividis.png"), io.imread(p + "/life_inred.png")
        self.cmaps = {'viridis': viridis,
                      'plasma': plasma,
                      'inferno': inferno,
                      'magma': magma,
                      'cividis': cividis,
                      'lifeinred_color': life_inred}
        self._list_colormap = ttk.Combobox(self._colormapPanel, values=list(self.cmaps.keys()), state="readonly")
        self._list_colormap.current(2)
        self._list_colormap.bind("<<ComboboxSelected>>", self._actualize)
        self._list_colormap.place(x=150, y=10)
        label_maps = tk.Label(self._colormapPanel, anchor='nw', text='Choose a colormap :', background='white')
        label_maps.place(x=5, y=10)
        self._colormap_selected = tk.Canvas(self._colormapPanel, width=width - 20, height=20, background='white')
        self._colormap_selected.place(x=5, y=50)
        self._colorbar_selected = prepare_image(self.cmaps[self._list_colormap.get()], size=(width - 20, 20))
        self._colormap_selected.create_image(round((width - 20) / 2), 10, image=self._colorbar_selected)

        #################################### COLOR DOMAIN BOX ##########################################
        color_domain = ['HSV', 'LAB']
        self._list_color_domain = ttk.Combobox(self._domainPanel, values=color_domain, state="readonly")
        self._list_color_domain.current(1)
        # self._list_color_domain.bind("<<ComboboxSelected>>", s)
        self._list_color_domain.place(x=150, y=10)
        label_domain = tk.Label(self._domainPanel, anchor='nw', text='Choose a fusion domain :', background='white')
        label_domain.place(x=5, y=10)

        #################################### GRADIENT BOX ##########################################
        ## Cascade list of filter
        filter_gradient = ['Prewitt', 'Sobel', 'Canny', 'Roberts', 'Perso']
        self._list_filter_gradient = ttk.Combobox(self._gradientPanel, values=filter_gradient, state="readonly")
        self._list_filter_gradient.current(0)
        # self._list_filter_gradient.bind("<<ComboboxSelected>>", self._filter_change)
        self._list_filter_gradient.place(x=150, y=10)
        label_filter = tk.Label(self._gradientPanel, anchor='nw', text='Choose a gradient filter :', background='white')
        label_filter.place(x=5, y=10)
        ## Frame for radiobuttons
        self._frame_edges = tk.LabelFrame(self._gradientPanel, text='Edges to compute', bg='white', width=180,
                                          height=110)
        self._frame_edges.grid_propagate(0)
        self._frame_edges.place(x=10, y=35)
        self._frame_image_edges = tk.LabelFrame(self._gradientPanel, text='Image support', bg='white', width=180,
                                                height=110)
        self._frame_image_edges.grid_propagate(0)
        self._frame_image_edges.place(x=245, y=35)

        ## radiobuttons for edges
        self.chosen_edges = tk.IntVar()
        self._IR_edges = tk.Radiobutton(self._frame_edges, variable=self.chosen_edges,
                                        text='Infrared Egdes',
                                        value=1, bg='white')
        self._IR_edges.place(x=5, y=5)
        self._VIS_edges = tk.Radiobutton(self._frame_edges, variable=self.chosen_edges,
                                         text='Visible Egdes',
                                         value=2, bg='white')
        self._VIS_edges.place(x=5, y=25)
        self._BOTH_edges = tk.Radiobutton(self._frame_edges, variable=self.chosen_edges,
                                          text='Visible & Infrared Egdes',
                                          value=3, bg='white')
        self._BOTH_edges.place(x=5, y=45)
        self._NO_edges = tk.Radiobutton(self._frame_edges, variable=self.chosen_edges,
                                        text='None',
                                        value=0, bg='white')
        self._NO_edges.place(x=5, y=65)

        ## radiobuttons for Image
        self.chosen_images = tk.IntVar()
        self._IR_image = tk.Radiobutton(self._frame_image_edges, variable=self.chosen_images,
                                        text='Infrared Image',
                                        value=1, bg='white')
        self._IR_image.place(x=5, y=5)
        self._VIS_image = tk.Radiobutton(self._frame_image_edges, variable=self.chosen_images,
                                         text='Visible Image',
                                         value=2, bg='white')
        self._VIS_image.place(x=5, y=25)
        self._BOTH_image = tk.Radiobutton(self._frame_image_edges, variable=self.chosen_images,
                                          text='Visible & Infrared Image',
                                          value=3, bg='white')
        self._BOTH_image.place(x=5, y=45)
        self._NO_image = tk.Radiobutton(self._frame_image_edges, variable=self.chosen_images,
                                        text='Only Edges',
                                        value=0, bg='white')
        self._NO_image.place(x=5, y=65)

        #################################### FUSION BOX ##########################################
        self._slide_ratio = tk.Scale(self._fusionPanel, from_=0, to=100, orient=tk.HORIZONTAL, length=width - 20,
                                     bg='white', activebackground='black')
        self._slide_ratio.place(x=5, y=10)
        self._slide_ratio.set(50)
        label_ratio = tk.Label(self._fusionPanel, anchor='nw', text='Visible', background='white')
        label_ratio.place(x=5, y=60)
        label_ratio = tk.Label(self._fusionPanel, anchor='nw', text='Infrared', background='white')
        label_ratio.place(x=width - 65, y=60)
        self._fusion_checkbox_choice = tk.IntVar()
        self._fusion_checkbox_choice.set(1)
        self._choice_domain = tk.Radiobutton(self._fusionPanel, text="Color Domain Fusion",
                                             command=self._fusion_method_selection,
                                             variable=self._fusion_checkbox_choice, value=1, bg='white')
        self._choice_colormap = tk.Radiobutton(self._fusionPanel, text="Colormap Fusion",
                                               command=self._fusion_method_selection,
                                               variable=self._fusion_checkbox_choice, value=2, bg='white')
        self._choice_domain.place(x=5, y=90)
        self._choice_colormap.place(x=5, y=115)
        self._fusion_method_selection()

    def _menu(self, master):
        menubar = tk.Menu(master)
        File = tk.Menu(menubar, tearoff=0)
        File.add_command(label="Snapshot", command=lambda: self._thread_manager(1, self._snapshot))
        File.add_separator()
        File.add_command(label="Quit", command=self._quit_app)
        menubar.add_cascade(label="File", menu=File)

        Flux_Visible = tk.Menu(menubar, tearoff=0)
        Flux_Visible.add_command(label="Open a flux IP Camera",
                                 command=lambda: self._open_a_channel(source=self.source))
        Flux_Visible.add_command(label="Open a flux webcam...", command=lambda: self._open_a_channel())
        Flux_Visible.add_separator()
        Flux_Visible.add_command(label="Stop the flux", command=lambda: self._destroy_camera('vis'))
        menubar.add_cascade(label="Open a visible stream", menu=Flux_Visible)

        Flux_Ir = tk.Menu(menubar, tearoff=0)
        # Flux_Ir.add_command(label="Open a flux IP Camera",
        #                          command=lambda: self._open_a_channel(source=self.source))
        Flux_Ir.add_command(label="Open a IR webcam...",
                            command=lambda: self._thread_manager(1, self._open_a_channel_ir))
        Flux_Ir.add_separator()
        Flux_Ir.add_command(label="Stop the flux", command=lambda: self._destroy_camera('ir'))
        menubar.add_cascade(label="Open an IR stream", menu=Flux_Ir)

        Calibration = tk.Menu(menubar, tearoff=0)
        Calibration.add_command(label='Orientation Estimation',
                                command=lambda: self._thread_manager(1, self._orientation_estimation))
        Calibration.add_command(label='Manual Homography', command=self._manual_homography)
        Calibration.add_separator()
        Calibration.add_command(label='Toggle Homography', command=self._toggleHomography)
        menubar.add_cascade(label="Calibration", menu=Calibration)

        self.res = tk.IntVar(value=0)
        # Res = tk.OptionMenu(master, self.res, 'Original Size', "640x480", "720x480", "1280x720", "1280x960", "1920x1080", command=self._change_window_size)
        Res = tk.Menu(menubar, tearoff=0)
        Res.add_radiobutton(label='Original Size', value=0, variable=self.res, command=self._change_window_size)
        Res.add_radiobutton(label="640x480", value=1, variable=self.res, command=self._change_window_size)
        Res.add_radiobutton(label="720x480", value=2, variable=self.res, command=self._change_window_size)
        Res.add_radiobutton(label="1280x720", value=3, variable=self.res, command=self._change_window_size)
        Res.add_radiobutton(label="1280x960", value=4, variable=self.res, command=self._change_window_size)
        Res.add_radiobutton(label="1920x1080", value=5, variable=self.res, command=self._change_window_size)
        menubar.add_cascade(label="Resolution", menu=Res)

        master.configure(menu=menubar)

    def _change_window_size(self):
        if self.res.get() == 0:
            self.Windowsize = 0
        elif self.res.get() == 1:
            self.Windowsize = (640, 480)
        elif self.res.get() == 2:
            self.Windowsize = (720, 480)
        elif self.res.get() == 3:
            self.Windowsize = (1280, 720)
        elif self.res.get() == 4:
            self.Windowsize = (1280, 960)
        elif self.res.get() == 5:
            self.Windowsize = (1920, 1080)

    def _toggleHomography(self):
        if self._calibration.get():
            self._calibration.set(0)
        else:
            self._calibration.set(1)

    def _fusion_method_selection(self):
        if self._fusion_checkbox_choice.get() == 1:
            enableChildren(self._domainPanel)
            disableChildren(self._colormapPanel)
        elif self._fusion_checkbox_choice.get() == 2:
            enableChildren(self._colormapPanel)
            disableChildren(self._domainPanel)

    def _enable_gradient(self):
        if self.camera_ir and self.camera_vis:
            enableChildren(self._frame_edges)
            enableChildren(self._frame_image_edges)
            self.chosen_edges.set(0)
            self.chosen_images.set(3)
        elif self.camera_ir:
            disableChildren(self._frame_edges)
            disableChildren(self._frame_image_edges)
            self._IR_edges.configure(state='normal')
            self._IR_image.configure(state='normal')
            self._NO_edges.configure(state='normal')
            self._NO_image.configure(state='normal')
            self.chosen_edges.set(0)
            self.chosen_images.set(1)
        elif self.camera_vis:
            disableChildren(self._frame_edges)
            disableChildren(self._frame_image_edges)
            self._VIS_edges.configure(state='normal')
            self._VIS_image.configure(state='normal')
            self._NO_edges.configure(state='normal')
            self._NO_image.configure(state='normal')
            self.chosen_edges.set(0)
            self.chosen_images.set(2)
        else:
            disableChildren(self._frame_edges)
            disableChildren(self._frame_image_edges)
            self.chosen_edges.set(0)
            self.chosen_images.set(0)

    def _open_a_channel_ir(self):
        ## Creation of the control panel for the IR stream
        if not self.camera_ir:
            self._create_IR_image_panel()
        if self.camera_ir:
            self.camera_ir.stop()
        try:
            self._connect_device()
        except lynred_py.error_t as e:
            print('lynred_py.error_t caught: {}'.format(e))
        except RuntimeError as e:
            print('RuntimeError caught: {}'.format(e))
        except Exception as e:
            print('exception caught: {}'.format(e))
        self._pre_process_pipe()
        self.camera_ir.start()
        # start a thread that constantly pools the video sensor for
        # the most recently read frame
        self._enable_gradient()
        self._thread_manager(self.thread, self.videoLoop)

    def _open_a_channel(self, source=None):
        if not self.camera_vis:
            self._create_VIS_image_panel()
        if self.camera_vis:
            self.camera_vis.stop()
        if source:
            try:
                self.camera_vis = VideoStream(source)
                self.camera_vis.start()
            except ValueError("The IP adress is not reachable"):
                self.camera_vis = VideoStream(0)
                self.camera_vis.start()
        else:
            self.camera_vis = VideoStream(0)
            self.camera_vis.start()
        # start a thread that constantly pools the video sensor for
        # the most recently read frame
        self._enable_gradient()
        self._thread_manager(self.thread, self.videoLoop)

    def _create_VIS_image_panel(self):
        self._frame_VIS_image = tk.LabelFrame(self.master, text='Visible Flux Panel', bg='white', width=400, height=200)
        self._frame_VIS_image.grid_propagate(0)
        if not self.camera_ir:
            self._frame_VIS_image.place(x=0, y=565)
        else:
            self._frame_VIS_image.place(x=0, y=770)
        ## Image processing frame
        self._frame_image_process_vis = tk.LabelFrame(self._frame_VIS_image, text='Image processing', bg='white',
                                                      width=390, height=100)
        self._frame_image_process_vis.grid_propagate(0)
        self._frame_image_process_vis.place(x=5, y=5)
        ## RadioButton for Histo improvment
        self._histo_choice_vis = tk.IntVar()
        self._histo_clahe_ir = tk.Radiobutton(self._frame_image_process_vis, variable=self._histo_choice_vis,
                                              text='CLAHE equalization', value=1, bg='white')
        self._histo_clahe_ir.place(x=5, y=5)
        self._histo_classic_ir = tk.Radiobutton(self._frame_image_process_vis, variable=self._histo_choice_vis,
                                                text='Classic Histo equalization', value=2, bg='white')
        self._histo_classic_ir.place(x=5, y=25)
        self._histo_None_ir = tk.Radiobutton(self._frame_image_process_vis, variable=self._histo_choice_vis,
                                             text='No equalization', value=0, bg='white')
        self._histo_None_ir.place(x=5, y=45)

    def _create_IR_image_panel(self):
        self._frame_IR_image = tk.LabelFrame(self.master, text='IR Flux Panel', bg='white', width=400, height=200)
        self._frame_IR_image.grid_propagate(0)
        if not self.camera_vis:
            self._frame_IR_image.place(x=0, y=565)
        else:
            self._frame_IR_image.place(x=0, y=770)
        ## Image processing frame
        self._frame_image_process_ir = tk.LabelFrame(self._frame_IR_image, text='Image processing', bg='white',
                                                     width=390, height=100)
        self._frame_image_process_ir.grid_propagate(0)
        self._frame_image_process_ir.place(x=5, y=5)
        ## RadioButton for Histo improvment
        self._histo_choice_ir = tk.IntVar()
        self._histo_clahe_ir = tk.Radiobutton(self._frame_image_process_ir, variable=self._histo_choice_ir,
                                              text='CLAHE equalization', value=1, bg='white')
        self._histo_clahe_ir.place(x=5, y=5)
        self._histo_classic_ir = tk.Radiobutton(self._frame_image_process_ir, variable=self._histo_choice_ir,
                                                text='Classic Histo equalization', value=2, bg='white')
        self._histo_classic_ir.place(x=5, y=25)
        self._histo_None_ir = tk.Radiobutton(self._frame_image_process_ir, variable=self._histo_choice_ir,
                                             text='No equalization', value=0, bg='white')
        self._histo_None_ir.place(x=5, y=45)
        self._enable_gradient()

    def _destroy_camera(self, type_camera):
        if type_camera == "vis":
            self.camera_vis.stop()
            self.camera_vis = None
            self._enable_gradient()
            self._frame_VIS_image.destroy()
            if self.camera_ir:
                self._frame_IR_image.place_configure(x=0, y=565)
        else:
            self.camera_ir.stop()
            self.camera_ir = None
            self._enable_gradient()
            self._frame_IR_image.destroy()
            if self.camera_vis:
                self._frame_VIS_image.place_configure(x=0, y=565)

    def _thread_manager(self, thread, func):
        if thread is self.thread:
            if self.thread:
                if not thread.is_alive():
                    self.thread = threading.Thread(target=func, args=())
                    self.thread.start()
            else:
                self.stopEvent = threading.Event()
                self.thread = threading.Thread(target=func, args=())
                self.thread.start()
        else:
            thread = threading.Thread(target=func, args=())
            thread.setDaemon(True)
            thread.start()

    def _connect_device(self):
        dalab_cameras = lynred_py.acq.discover_device_alab_cameras()
        for camera_name in dalab_cameras:
            print("Connecting to the Device Alab camera " + camera_name.c_str())
            self.camera_ir = lynred_py.acq.create_camera_device_alab(camera_name.c_str())

    def _actualize(self, event=None):
        self._colorbar_selected = prepare_image(self.cmaps[self._list_colormap.get()], size=(400 - 20, 20))
        self._colormap_selected.create_image(round((400 - 20) / 2), 10, image=self._colorbar_selected)

    def videoLoop(self):
        # DISCLAIMER:
        # I'm not a GUI developer, nor do I even pretend to be. This
        # try/except statement is a pretty ugly hack to get around
        # a RunTime error that Tkinter throws due to threading
        cv.namedWindow('Fusion Window')
        k = 0
        fps = []
        checkpoint = []
        try:
            # keep looping over frames until we are instructed to stop
            while not self.stopEvent.is_set() and (self.camera_ir or self.camera_vis):
                if k == 0:
                    fps = []
                    checkpoint = []
                k += 1
                if isinstance(self.Windowsize, tuple):
                    size = self.Windowsize
                    ratio = size[1] / size[0]
                else:
                    ratio = 1
                    size = 0
                # Start time counter
                start = time.time()
                # start2 = time.time()
                # Start the selected camera(s) ############################################################
                # Only Visible
                if self.camera_vis and not self.camera_ir:
                    if self._acquire_frame_vis(ratio, size):
                        break
                    self.FUS = None
                    self.frame_vis = self._process_pipe(self.frame_vis)
                    if size:
                        if self.frame_vis.shape[:2] != (size[1], size[0]):
                            self.frame_vis = crop_image(self.frame_vis, ratio=ratio)
                            self.frame_vis = cv.resize(self.frame_vis, size)
                    cv.imshow('Fusion Window', self.frame_vis)
                # Only Infrared
                elif self.camera_ir and not self.camera_vis:
                    if self._acquire_frame_ir():
                        break
                    self.FUS = None
                    self.frame_ir = self._process_pipe(self.frame_ir)
                    if size:
                        if self.frame_ir.shape[:2] != (size[1], size[0]):
                            self.frame_ir = crop_image(self.frame_ir, ratio=ratio)
                            self.frame_ir = cv.resize(self.frame_ir, size)
                    cv.imshow('Fusion Window', self.frame_ir)
                # Fusion if there are two cameras
                elif self.camera_ir and self.camera_vis:
                    ratio_fusion = self._slide_ratio.get() / 100
                    if self._acquire_frame_vis(ratio, size):
                        break
                    if self._acquire_frame_ir():
                        break
                    self.frame_ir, self.frame_vis = size_matcher(self.frame_ir, self.frame_vis)
                    self.frame_ir = self._process_pipe(self.frame_ir)
                    self.frame_vis = self._process_pipe(self.frame_vis)
                    frame_vis_cut = self.frame_vis[self._center[0]:self._center[0] + self.frame_ir.shape[0],
                                    self._center[1]:self._center[1] + self.frame_ir.shape[1]]
                    self.FUS = self.frame_vis.RGB()
                    if self._fusion_checkbox_choice.get() == 1:
                        self.FUS[self._center[0]:self._center[0] + self.frame_ir.shape[0],
                        self._center[1]:self._center[1] + self.frame_ir.shape[1]] = \
                            grayscale_fusion(self.frame_ir, frame_vis_cut,
                                             ratio=ratio_fusion, domain=self._list_color_domain.get())
                    elif self._fusion_checkbox_choice.get() == 2:
                        self.FUS[self._center[0]:self._center[0] + self.frame_ir.shape[0],
                        self._center[1]:self._center[1] + self.frame_ir.shape[1]] = \
                            colormap_fusion2(self.frame_ir, frame_vis_cut,
                                             ratio=ratio_fusion, colormap=self._list_colormap.get())
                    if size:
                        if self.FUS.shape[:2] != (size[1], size[0]):
                            self.FUS = crop_image(self.FUS, ratio=ratio)
                            self.FUS = ImageCustom(cv.resize(self.FUS, size))
                    cv.imshow('Fusion Window', self.FUS.BGR())

                key = cv.waitKey(1)
                if key == 27:  # exit on ESC break
                    if self.camera_ir:
                        self._destroy_camera('ir')
                    if self.camera_vis:
                        self._destroy_camera('vis')
                    break
                fps.append(round(1 / (time.time() - start)))
                if k == 100:
                    k = 0
                    fps = np.mean(fps)
                    # checkpoint = np.mean(checkpoint)
                    print(f"{fps} fps")
                    # f"Operation duration : {checkpoint} sec")
            print("end of the loop")
            cv.destroyAllWindows()
        except RuntimeError:
            print("[INFO] caught a RuntimeError")

    def _snapshot(self):
        t = strftime("%d_%b_%H_%M_%S", time.localtime())
        if self.camera_ir or self.camera_vis:
            p = join(abspath(dirname('Snapshot')), '..', 'Snapshot')
            number = round(len(listdir(p)) / 2 + 0.4)
            print(number)
            ir_name = 'IR_' + t + '.jpg'
            rgb_name = 'VIS_' + t + '.jpg'
            ir_path = join(p, ir_name)
            rgb_path = join(p, rgb_name)
            if self.camera_ir and not self.camera_vis:
                dis = self.frame_ir
                io.imsave(ir_path, self.frame_ir)
            elif self.camera_vis and not self.camera_ir:
                dis = cv.cvtColor(self.frame_vis, cv.COLOR_BGR2RGB)
                io.imsave(rgb_path, cv.cvtColor(self.frame_vis, cv.COLOR_BGR2RGB))
            else:
                io.imsave(ir_path, self.frame_ir)
                d1, d2 = round((self.frame_vis.shape[0] - 240) / 2), round((self.frame_vis.shape[1] - 320) / 2)
                io.imsave(rgb_path, cv.cvtColor(self.frame_vis[d1:-d1, d2:-d2], cv.COLOR_BGR2RGB))
                ir = cv.resize(self.frame_ir, (self.frame_vis.shape[1], self.frame_vis.shape[0]))
                ir = np.transpose(np.array((ir, ir, ir)), (1, 2, 0))
                dis = np.hstack((cv.cvtColor(self.frame_vis, cv.COLOR_BGR2RGB), ir))
            fig = plt.figure()
            plt.imshow(dis)
            plt.xticks([])
            plt.yticks([])
            fig.subplots_adjust(top=1, bottom=0,
                                left=0, right=1)
            plt.show()
        else:
            tk.messagebox.askquestion("No input", "No input to save", icon="warning")

    def _acquire_frame_vis(self, ratio, size):
        frame = self.camera_vis.read()
        if frame is None:
            print('problem Input Visible')
            return 1
        self.frame_vis = frame
        m, n = self.frame_vis.shape[:2]
        if size:
            if m / n != ratio:
                self.frame_vis = crop_image(self.frame_vis, ratio=ratio)
            if self.frame_vis.shape[0] > size[1] or self.frame_vis.shape[1] > size[0]:
                self.frame_vis = cv.resize(self.frame_vis, size)
        self.frame_vis = ImageCustom(self.frame_vis)
        self.frame_vis.cmap = 'BGR'

    def _acquire_frame_ir(self):
        image = lynred_py.base_t.image_t()
        self.camera_ir.get_image(image)
        if image.empty():
            raise lynred_py.error_t("Error while getting test image")
        self.pipe.execute(image, image)
        self.frame_ir = ImageCustom(np.asarray(image))
        if self.frame_ir is None:
            print('problem Input Infrarouge')
            return 1

    def _process_pipe(self, image):
        if image.info == 'Visible':
            if self._histo_choice_vis.get():
                # Histo equalisation
                lab = image.LAB()
                temp = ImageCustom(lab[:, :, 0])
                temp = histogram_equalization(temp, method=self._histo_choice_vis.get())
                lab[:, :, 0] = temp
                image = lab.BGR()
        if image.info == 'IR':
            # Histo equalisation
            image = histogram_equalization(image, method=self._histo_choice_ir.get())
            if self._calibration.get():
                image = ImageCustom(cv.warpPerspective(image, self._homography_matrix, (320, 240)))
        return image

    def _pre_process_pipe(self):
        pipe = lynred_py.algo.pipe_shutterless_2ref_t()
        pipe.load_from_file("D:\Travail\LYNRED\FUSION\interface\data_interface/test.spbin")
        self.pipe = pipe

    def _orientation_estimation(self):
        if self.camera_ir and self.camera_vis:
            tot = np.ones([self.frame_vis.shape[0] - self.frame_ir.shape[0] + 2,
                           self.frame_vis.shape[1] - self.frame_ir.shape[1] + 2])
            for i in range(1):
                self._angle, self._gap, self._center, tot = orientation_calibration_cam(self.frame_ir, self.frame_vis,
                                                                                        method='prewitt', tot=tot)
            p = join(abspath(dirname('tools')), '..', 'tools', 'calibration_selection')
            with open(join(p, 'angle_calibration_cam'), "wb") as f:
                pickle.dump(self._angle, f)
            with open(join(p, 'length_calibration_cam'), "wb") as f:
                pickle.dump(self._gap, f)
            with open(join(p, 'center_calibration_cam'), "wb") as f:
                pickle.dump(self._center, f)
        else:
            print('The two video flux are required to do the calibration')
        print(f"Angle of the entraxe from horizontal : {self._angle}\n"
              f"Range of the spread in pixels : {self._gap}\n"
              f"Position of the main object : {self._center}")

    def _manual_homography(self):
        p = join(abspath(dirname('Snapshot')), '..', 'tools', 'calibration_selection')
        image_ir = ImageCustom(join(p, 'IR_homography.jpg'))
        image_vis = ImageCustom(join(p, 'VIS_homography.jpg'))
        # image_vis = ImageCustom(cv.resize(image_vis, (image_ir.shape[1], image_ir.shape[0])))
        _, _, tform = manual_calibration(image_ir, image_vis)
        with open(join(p, "transform_matrix_cam"), "wb") as p:
            pickle.dump(tform, p)
        self._homography_matrix = self._load_homography()

    def _load_homography(self):
        p = join(abspath(dirname('tools')), '..', 'tools', 'calibration_selection')
        try:
            if os.stat(join(p, 'transform_matrix_cam')).st_size > 0:
                with open(join(p, 'transform_matrix_cam'), "rb") as f:
                    homography_matrix = pickle.load(f)
            else:
                homography_matrix = None
        except OSError:
            print("There is no Transformation matrix there !")
            homography_matrix = None
        return homography_matrix
