"""
Created on Tue Aug  3 11:22:02 2021

@author: graceluo

Construct setup frame for bnp_gui
"""
#!/home/beams/USERBNP/.conda/envs/py36/bin/python

import tkinter as tk
from tkinter import ttk
import os, h5py, time
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib import colors, patches
from pvComm import pvComm
from misc import coordinate_transform, checkEntryDigit, limit_stringvar_length
import tifffile
from mic_vis.bnp.mda import get_mda_positioners


class setupFrame:
    def Display2Data(self, Axe, x, y):
        return Axe.transData.inverted().transform(np.array([(x, y)]))[0]

    def choose_xrf_folder(self):
        if self.pvComm.getDir() is not None:
            initialDir = os.path.join(self.pvComm.getDir(), "img.dat")
            initialDir = (
                initialDir if os.path.exists(initialDir) else self.pvComm.getDir()
            )
        else:
            initialDir = "/mnt/micdata1/2idd"

        h5_folder = tk.filedialog.askdirectory(initialdir=initialDir)
        self.xrf_folder.set(h5_folder)
        ext = ".h5"
        if len(h5_folder) > 10:
            self.xrf_file_combobox["values"] = [
                i for i in os.listdir(h5_folder) if i[-len(ext) :] == ext
            ]
        else:
            h5_folder = tk.filedialog.askdirectory(
                initialdir="/mnt/micdata1/bnp/2021-3/"
            )

    def choose_ptycho_folder(self):
        
        if len(self.ptycho_folder.get()) < 5:
            if self.pvComm.getDir() is not None:
                ptychoDir = os.path.join(self.pvComm.getDir(), "ptychi_recons")
                ptychoDir = (
                    ptychoDir if os.path.exists(ptychoDir) else self.pvComm.getDir()
                )
            else:
                ptychoDir = "/mnt/micdata1/bnp/2021-3/"
        else:
            ptychoDir = self.ptycho_folder.get()

        ptycho_folder = tk.filedialog.askdirectory(initialdir=ptychoDir)

        if len(ptycho_folder) > 10:
            self.ptycho_folder.set(ptycho_folder)
            self.ptycho_scannum_combobox["values"] = [
                i for i in os.listdir(ptycho_folder) if os.path.isdir(os.path.join(ptycho_folder, i))
            ]

            self.ptycho_scannum_combobox.set("")
            self.ptycho_recon_combobox.set("")
            self.ptycho_roi_combobox.set("")
            self.ptycho_niter_combobox.set("")
            # self.ptycho_recon_combobox["values"] = []
            self.openfilemsg.set(f"Ptycho selected path: {ptycho_folder}")
            self.open_msg_label.config(fg="green")
        else:
            self.ptycho_folder.set(ptychoDir)
        
    def list_recon_directory(self, *args):
        
        # Initialize the comboboxes
        self.ptycho_roi_combobox["values"] = []
        self.ptycho_recon_combobox["values"] = []
        self.ptycho_niter_combobox["values"] = []
        self.ptycho_roi_combobox.set("")
        self.ptycho_recon_combobox.set("")
        self.ptycho_niter_combobox.set("")

        recon_methods = []
        selected_folder = os.path.join(self.ptycho_folder.get(), self.ptycho_scannum_combobox.get())
        if os.path.exists(selected_folder):
            found_folders = [i for i in os.listdir(selected_folder) if os.path.isdir(os.path.join(selected_folder, i))]
            if "ML_recon" in selected_folder:
                # This considers recon by ptychoshelf and will need to update the roi combobox
                self.ptychirecon_dir_selected = False
                show_type = "O_phase_roi"
                self.ptycho_roi_combobox["values"] = found_folders
                if len(self.ptycho_roi_combobox["values"]) > 0:
                    self.ptycho_roi_combobox.current(0)
                    roi_path = os.path.join(selected_folder, self.ptycho_roi_combobox.get())
                    recon_methods = [i for i in os.listdir(roi_path) if os.path.isdir(os.path.join(roi_path, i))]
                
            else:
                # This considers recon by Ptychi
                self.ptychirecon_dir_selected = True
                recon_methods = found_folders
                show_type = "object_ph"


            self.ptycho_recon_combobox["values"] = recon_methods
            if len(self.ptycho_recon_combobox["values"]) > 0:
                self.ptycho_recon_combobox.current(0)

            # Lets get the niter list
            result_folder = os.path.join(selected_folder,
                                         self.ptycho_roi_combobox.get(),
                                         self.ptycho_recon_combobox.get(),
                                         show_type)
                                         
            print(f"result_folder: {result_folder}")
            if os.path.exists(result_folder):
                ext = ".tiff"
                niter_list = [i for i in os.listdir(result_folder) if i[-len(ext) :] == ext]
                niter_list.sort(key=lambda x: int(x.split(".tiff")[0].replace(show_type+"_Niter", "")))
                self.ptycho_niter_combobox["values"] = niter_list
                self.ptycho_niter_combobox.current(0)
                # try:
                self._load_ptycho_scan()
                # except:
                #     self.openfilemsg.set("Having trouble opening %s" % self.tiff_filename)
                #     self.open_msg_label.config(fg="red")


                #     self.openfilemsg.set(f"Ptycho selected path: {selected_folder}")
                #     self.open_msg_label.config(fg="green")
                # else:
                #     self.openfilemsg.set(f"Folder is empty: {selected_folder}")
                #     self.open_msg_label.config(fg="red")
        # else:
        #     self.openfilemsg.set(f"The folder is not found: {selected_folder}")
        #     self.open_msg_label.config(fg="red")
        
    def list_niter_results(self, *args):
        print(f"list_niter_results: {args}")
        show_type = "object_ph"
        selected_folder = os.path.join(self.ptycho_folder.get(), 
                                       self.ptycho_scannum_combobox.get(), 
                                       self.ptycho_roi_combobox.get(),
                                       self.ptycho_recon_combobox.get(), 
                                       show_type)
        print(selected_folder)
        ext = ".tiff"
        if os.path.exists(selected_folder):
            niter_list = [i for i in os.listdir(selected_folder) if i[-len(ext) :] == ext]
            niter_list.sort(key=lambda x: int(x.split(".tiff")[0].replace(show_type+"_Niter", "")))
            self.ptycho_niter_combobox["values"] = [i for i in os.listdir(selected_folder) if i[-len(ext) :] == ext]
            self.ptycho_niter_combobox.current(0)

    def updateFileList(self):
        ext = ".h5"
        self.xrf_file_combobox["values"] = [
            i for i in os.listdir(self.xrf_folder.get()) if i[-len(ext) :] == ext
        ]

    def _update_ptycho_mda_filepath(self):
        
        if self.ptychirecon_dir_selected:
            scan_num = int(self.ptycho_scannum_combobox.get().replace("S", ""))
            mda_dir = os.path.join(self.ptycho_folder.get().split("/ptychi_recons")[0], "mda")
        else:
            scan_num = int(self.ptycho_scannum_combobox.get().replace("fly", ""))
            mda_dir = os.path.join(self.ptycho_folder.get().split("/results/ML_recon")[0], "mda")

        # Lets find the mda file that contains scan number in the mda directory
        mda_files = [i for i in os.listdir(mda_dir) if i.endswith(".mda")]
        for mda_file in mda_files:
            if str(scan_num) in mda_file:
                self.mda_filepath = os.path.join(mda_dir, mda_file)
                break
        else:
            self.mda_filepath = None
            self.openfilemsg.set("MDA file not found for scan number %s" % scan_num)
            self.open_msg_label.config(fg="red")


    def _update_ptycho_tiff_filepath(self):
        if self.ptychirecon_dir_selected:
            show_type = "object_ph"
        else:
            show_type = "O_phase_roi"
        self.tiff_filename = os.path.join(self.ptycho_folder.get(), 
                                    self.ptycho_scannum_combobox.get(),
                                    self.ptycho_roi_combobox.get(),
                                    self.ptycho_recon_combobox.get(), 
                                    show_type, 
                                    self.ptycho_niter_combobox.get())
            

    def _load_ptycho_scan(self):
        self._update_ptycho_tiff_filepath()
        self._update_ptycho_mda_filepath()
        self.tiff = tifffile.imread(self.tiff_filename)

        if self.tiff.dtype.kind == "u":
            vmax = np.iinfo(self.tiff.dtype).max
            norm = self.tiff.astype(np.float32) / vmax
            self.tiff = norm
            print(f"dtype of tiff is {self.tiff.dtype}")
            print(f"vmax: {self.tiff.max()}")
            print(f"vmin: {self.tiff.min()}")

        if self.mda_filepath is not None:
            positioners = get_mda_positioners(self.mda_filepath)
            self.y = np.linspace(positioners["y_pos"][1], positioners["y_pos"][-1], self.tiff.shape[0])
            self.x = np.linspace(positioners["x_pos"][0], positioners["x_pos"][-1], self.tiff.shape[1])
            self.file_z = positioners["z_pos"]
            self.file_theta = positioners["theta_pos"]
        
            self.Image2D = self.Axe2D.imshow(
                self.tiff,
                aspect="equal",
                interpolation="nearest",
                cmap="gray",
                origin="lower",
                norm=colors.SymLogNorm(linthresh=0.5)
                if self.log_button.config("relief")[-1] == "sunken"
                else colors.Normalize(),
            )
            self.Canvas2D.draw()

            if self.file_z is None:
                self.openfilemsg.set("Samz PV not found %s" % self.mda_filepath)
                self.open_msg_label.config(fg="red")
            else:
                msg = f"{self.ptycho_scannum_combobox.get()} - "
                msg += f"{self.ptycho_roi_combobox.get()} - "
                msg += f"{self.ptycho_niter_combobox.get()} is open"
                self.openfilemsg.set(msg)
                self.open_msg_label.config(fg="green")
        else:
            self.openfilemsg.set("Having trouble opening %s" % self.mda_filepath)
            self.open_msg_label.config(fg="red")

    def load_ptycho_scan(self, *args):
        self._update_ptycho_tiff_filepath()

        try:
            self.ShowROI_Rectangle.remove()
        except (AttributeError, ValueError):
            pass

        t_diff = 10
        fmtime = os.path.getmtime(self.tiff_filename)
        ctime = time.time()
        if (ctime - fmtime) >= t_diff:
            try:
                self._load_ptycho_scan()
            except:
                self.openfilemsg.set("Having trouble opening %s" % self.tiff_filename)
                self.open_msg_label.config(fg="red")
        else:
            self.openfilemsg.set("File not ready, getting update from other process")
            self.open_msg_label.config(fg="red")

    def load_xrf_scan(self, *args):
        self.h5_filename = os.path.join(self.xrf_folder.get(), self.xrf_file_combobox.get())
        try:
            self.h5.close()
        except:
            pass

        try:
            self.ShowROI_Rectangle.remove()
        except (AttributeError, ValueError):
            pass

        t_diff = 10
        fmtime = os.path.getmtime(self.h5_filename)
        ctime = time.time()
        if (ctime - fmtime) >= t_diff:
            try:
                self.h5 = h5py.File(self.h5_filename, "r")
                dets = []
                i_det = max(0, self.detector_combobox.current())
                dets = (
                    self.h5["/MAPS/channel_names"][:].astype(str).tolist()
                    + self.h5["/MAPS/scaler_names"][:].astype(str).tolist()
                )
                elmScalers = np.vstack(
                    (self.h5["/MAPS/XRF_roi_plus"][:], self.h5["/MAPS/scalers"][:])
                ) # changed to _plus on 3/13/2026
                self.detector_combobox["values"] = dets
                self.detector_combobox.current(i_det)
                self.x = self.h5["/MAPS/x_axis"][()]
                self.y = self.h5["/MAPS/y_axis"][()]
                pvlist = self.h5["/MAPS/extra_pvs"][0].astype(str).tolist()
                pvval = self.h5["/MAPS/extra_pvs"][1].astype(str).tolist()

                print(f"before self.detectors: {dets}")

                print(f"pvlist: {pvlist}")
                print(f"pvvalue: {pvval}")


                if len(pvlist) < 5:
                    self.file_z = None
                    self.file_theta = None
                
                else:
                    try:
                        self.file_z = float(
                            pvval[pvlist.index(self.pvComm.pvs["z_value_Act"].pv.pvname)]
                        )
                        self.file_theta = float(
                            pvval[pvlist.index(self.pvComm.pvs["sm_rot_Act"].pv.pvname)]
                        )
                    except:
                        print("Not able to get sample-z and theta from .h5... getting it from mda")
                        mda_file = os.path.join(self.xrf_folder.get().replace("img.dat", "mda"), self.xrf_file_combobox.get().split(".h5")[0])
                        mda_positions = get_mda_positioners(mda_file, get_z = True, get_theta = True)
                        self.file_z = mda_positions['z_pos']
                        self.file_theta = mda_positions['theta_pos']
                


                self.Image2D = self.Axe2D.imshow(
                    elmScalers[i_det],
                    aspect="equal",
                    interpolation="nearest",
                    cmap="inferno",
                    origin="lower",
                    norm=colors.SymLogNorm(linthresh=0.5)
                    if self.log_button.config("relief")[-1] == "sunken"
                    else colors.Normalize(),
                )
                self.Canvas2D.draw()
                if self.file_z is None:
                    self.openfilemsg.set("Samz PV not found %s" % self.h5_filename)
                    self.open_msg_label.config(fg="red")
                else:
                    self.openfilemsg.set("%s is open" % self.h5_filename)
                    self.open_msg_label.config(fg="green")
            except:
                self.openfilemsg.set("Having trouble opening %s" % self.h5_filename)
                self.open_msg_label.config(fg="red")
        else:
            self.openfilemsg.set("File not ready, getting update from other process")
            self.open_msg_label.config(fg="red")

    def plot_data(self, *args):
        i_det = self.detector_combobox.current()
        elmScalers = np.vstack(
            (self.h5["/MAPS/XRF_roi_plus"][:], self.h5["/MAPS/scalers"][:])
        )
        # change to XRF_roi_plus on 3/13/2026

        img_data = self.Image2D.get_array()
        plot_data = np.array(elmScalers[i_det])

        if img_data.shape != plot_data.shape:
            print("image data and plot data have different shapes and will need to replot")
            self.x = self.h5["/MAPS/x_axis"][()]
            self.y = self.h5["/MAPS/y_axis"][()]
            self.Image2D = self.Axe2D.imshow(
            plot_data,
            aspect="equal",
            interpolation="nearest",
            cmap="inferno",
            origin="lower",
            norm=colors.SymLogNorm(linthresh=0.5)
            if self.log_button.config("relief")[-1] == "sunken"
            else colors.Normalize(),
            )

        else:
            self.Image2D.set_array(plot_data)
            self.Image2D.set_cmap("inferno")
            if self.log_button.config("relief")[-1] == "sunken":
                # self.Image2D.set_norm(colors.LogNorm())
                self.Image2D.set_norm(colors.SymLogNorm(linthresh=0.5))
            else:
                self.Image2D.set_norm(colors.Normalize())

        self.openfilemsg.set("%s is open" % self.h5_filename)
        self.open_msg_label.config(fg="green")
        self.Canvas2D.draw()

    def logscale_changed(self):
        if self.log_button.config("relief")[-1] == "sunken":
            self.log_button.config(relief="raised")
            self.Image2D.set_norm(colors.Normalize())
        else:
            self.log_button.config(relief="sunken")
            self.Image2D.set_norm(colors.SymLogNorm(linthresh=0.5))
        self.Canvas2D.draw()

    def Draw_Rectangle(self, xmin, xmax, ymin, ymax, color, lw=2, animated=False):
        Rectangle = patches.Rectangle(
            (xmin, ymin),
            width=xmax - xmin,
            height=ymax - ymin,
            alpha=1,
            edgecolor="w",
            fill=False,
            linewidth=lw,
            animated=animated,
        )
        self.Axe2D.add_patch(Rectangle)
        return Rectangle


    def Canvas2D_Button_Pressed(self, event):
        self.xstart, self.ystart = list(
            map(
                lambda x: int(round(x, 0)),
                (self.Display2Data(self.Axe2D, event.x, event.y)),
            )
        )
        if event.button == 1:
            self.xycorr.set(
                "x, y: (%.2f, %.2f)" % (self.x[self.xstart], self.y[self.ystart])
            )
        if (event.button == 1) & (self.insertType.get() == "XY Center"):
            try:
                self.ShowROI_Rectangle.remove()
            except (AttributeError, ValueError):
                pass
            self.updateXYcenter()
        elif (event.button == 3) & (self.insertType.get() == "ScanBox"):
            self.Rectangle_Drawing = True
            try:
                self.ShowROI_Rectangle.remove()
            except (AttributeError, ValueError):
                pass
            self.Canvas2D.mpl_disconnect(self.Canvas2D_Button_Press_Event)
            self.Canvas2D_Button_Release_Event = self.Canvas2D.mpl_connect(
                "button_release_event", self.Canvas2D_Button_Released
            )

    def Canvas2D_Button_Released(self, event):
        self.xend, self.yend = list(
            map(
                lambda x: int(round(x, 0)),
                (self.Display2Data(self.Axe2D, event.x, event.y)),
            )
        )

        self.Canvas2D.mpl_disconnect(self.Canvas2D_Button_Release_Event)
        self.Canvas2D_Button_Press_Event = self.Canvas2D.mpl_connect(
            "button_press_event", self.Canvas2D_Button_Pressed
        )

        if self.Rectangle_Drawing:
            self.Rectangle_Drawing = False
            self.non_animated_background = None
            try:
                self.ShowROI_Rectangle.remove()
            except (AttributeError, ValueError):
                pass
            xmin = int(min(self.xend, self.xstart))
            xmax = int(max(self.xend, self.xstart))
            ymin = int(min(self.yend, self.ystart))
            ymax = int(max(self.yend, self.ystart))
            self.ShowROI_Rectangle = self.Draw_Rectangle(xmin, xmax, ymin, ymax, "w")
            self.updateScanBoxParms()

    def Canvas2D_Mouser_Hover(self, event):
        self.xend, self.yend = list(
            map(
                lambda x: int(round(x, 0)),
                (self.Display2Data(self.Axe2D, event.x, event.y)),
            )
        )
        try:
            dim_y, dim_x = self.Image2D.get_array().shape
        except:
            return
        if (
            (self.yend < dim_y - 0.5)
            and (self.yend > -0.5)
            and (self.xend > -0.5)
            and (self.xend < dim_x - 0.5)
        ):
            if self.Rectangle_Drawing:
                if self.non_animated_background != None:
                    # restore the clean slate background
                    self.Canvas2D.restore_region(self.non_animated_background)
                    if self.xstart > self.xend:  # modify the starting point
                        self.Rectangle.set_x(self.xend)
                    self.Rectangle.set_width(abs(self.xend - self.xstart))
                    if self.ystart > self.yend:  # modify the starting point
                        self.Rectangle.set_y(self.yend)
                    self.Rectangle.set_height(abs(self.yend - self.ystart))
                    self.Axe2D.draw_artist(self.Rectangle)
                    self.Canvas2D.blit(self.Axe2D.bbox)
                else:
                    xmax = int(max(self.xend, self.xstart))
                    xmin = int(min(self.xend, self.xstart))
                    ymax = int(max(self.yend, self.ystart))
                    ymin = int(min(self.yend, self.ystart))
                    self.Rectangle = self.Draw_Rectangle(
                        xmin, xmax, ymin, ymax, "w", animated=True
                    )
                    self.Canvas2D.draw()
                    self.non_animated_background = self.Canvas2D.copy_from_bbox(
                        self.Axe2D.bbox
                    )

    def updateXYcenter(self):
        x_scan = np.round(self.x[self.xstart], 2)
        y_scan = np.round(self.y[self.ystart], 2)
        z_scan = np.round(self.file_z, 2)
        target_theta = self.file_theta

        slabel = ["x_scan", "y_scan", "z_scan", "target_theta"]
        for s in slabel:
            self.scanParms[s].delete(0, tk.END)
            self.scanParms[s].insert(0, "%.2f" % (eval(s)))

        slabel = ["x_theta0", "y_theta0", "z_theta0"]
        if (self.file_theta ** 2) < 1e-3:
            svlabel = ["x_scan", "y_scan", "z_scan"]
        else:
            e = ""
            svlabel = ["e"] * 3

        for s_, sv_ in zip(slabel, svlabel):
            self.scanParms[s_].delete(0, tk.END)
            self.scanParms[s_].insert(0, "%s" % (str(eval(sv_))))

    def updateScanBoxParms(self):
        x_scan = np.round((self.x[self.xstart] + self.x[self.xend]) / 2, 2)
        y_scan = np.round((self.y[self.ystart] + self.y[self.yend]) / 2, 2)
        width = abs(self.x[self.xstart] - self.x[self.xend])
        height = abs(self.y[self.ystart] - self.y[self.yend])
        if self.file_z is not None:    
            z_scan = np.round(self.file_z, 2)
            target_theta = self.file_theta
            slabel = ["x_scan", "y_scan", "width", "height", "z_scan", "target_theta"]
            for s in slabel:
                self.scanParms[s].delete(0, tk.END)
                self.scanParms[s].insert(0, "%.2f" % (eval(s)))
    
            slabel = ["x_theta0", "y_theta0", "z_theta0"]
            if (self.file_theta ** 2) < 1e-3:
                svlabel = ["x_scan", "y_scan", "z_scan"]
            else:
                e = ""
                svlabel = ["e"] * 3
    
            for s_, sv_ in zip(slabel, svlabel):
                self.scanParms[s_].delete(0, tk.END)
                self.scanParms[s_].insert(0, "%s" % (str(eval(sv_))))
        else:
            slabel = ["x_scan", "y_scan", "width", "height"]
            for s in slabel:
                self.scanParms[s].delete(0, tk.END)
                self.scanParms[s].insert(0, "%.2f" % (eval(s)))
            flabel = ['z_scan', 'target_theta', 'x_theta0', 'y_theta0', 'z_theta0']
            for f in flabel: 
                self.scanParms[f].delete(0, tk.END)

    def calcTime(self):
        strlist = ["width", "height", "w_step", "h_step", "dwell"]
        try:
            value = [float(self.scanParms[s].get()) for s in strlist]
        except:
            value = [0] * len(strlist)

        if value[0] > 80:
            self.calctime.set("Width can not be bigger than 80")
            self.calctime_out.config(fg="red")
        elif 0 not in value:
            eta_ms = value[-1] * value[0] * value[1] / value[2] / value[3]
            eta_min = eta_ms / 1e3 / 60 / 0.8
            self.calctime.set("%.3f" % (eta_min))
            self.calctime_out.config(fg="green")


    def getUserDir(self):
        self.pardir.set(self.pvComm.getDir())

    def addToScanBtn(self, func):
        add_scan_btn = tk.Button(
            self.setupfrm, text="Add to scan list", command=func, width=50
        )
        add_scan_btn.grid(row=self.row + 5, column=self.col, columnspan=4)

    def updatexyz0(self):
        if (self.pvComm.getSMAngle() ** 2) < 1e-3:
            self.fillxyz0(empty=False)
            self.coordtform()
            self.updatexyz_msg.set(
                "x-, y-, z- theta0 updated"
                + "\n"
                + "Coordinate transformed. x-, y-, z- scan values updated"
            )
            self.updatexyz_label.config(fg="green")
        else:
            self.updatexyz_msg.set("Theta (Rotation) motor is not at 0")
            self.updatexyz_label.config(fg="red")
            
    def fillxyz0(self, empty = True):
        if empty:
            x,y,z = ['','','']
        else:
            x, y, z = self.pvComm.getXYZcenter()
        for a_, v_ in zip(["x_theta0", "y_theta0", "z_theta0"], [x, y, z]):
            self.scanParms[a_].delete(0, tk.END)
            self.scanParms[a_].insert(0, str(v_)) 
            
    def fillxyzScan(self, empty = True):
        if empty:
            x,y,z,theta = ['','','', '']
        else:
            x, y, z = self.pvComm.getXYZcenter()
            theta = self.pvComm.getSMAngle()
        for a_, v_ in zip(['x_scan', 'y_scan', 'z_scan', 'target_theta'], [x, y, z, theta]):
            self.scanParms[a_].delete(0, tk.END)
            self.scanParms[a_].insert(0, str(v_)) 
    
    def updatexyz(self):
        if (self.pvComm.getSMAngle() ** 2) < 1e-3:
            self.fillxyz0(empty=False)
            self.fillxyzScan()
        else:
            self.fillxyz0()
            self.fillxyzScan(empty=False)


    def coordtform(self):
        fields = ["target_theta", "x_theta0", "y_theta0", "z_theta0"]
        fvals = [float(self.scanParms[a_].get()) for a_ in fields]
        ctform = coordinate_transform(*fvals)
        writeFields = ["x_scan", "y_scan", "z_scan"]
        wfv = [round(ctform[a], 2) for a in ["x", "y", "z"]]
        for a_, v_ in zip(writeFields, wfv):
            self.scanParms[a_].delete(0, tk.END)
            self.scanParms[a_].insert(0, str(v_))
        self.updatexyz_msg.set("Coordinate transformed. x-, y-, z- scan values updated")
        self.updatexyz_label.config(fg="green")

    def __init__(self, tabControl):
        # =============================================================================
        # INITIALIZATION
        # =============================================================================
        self.setupfrm = ttk.Frame(tabControl)
        self.pvComm = pvComm()

        # Initialize variables
        self.file_theta = 0
        self.file_z = 0
        self.Rectangle_Drawing = False
        self.Dot_Drawing = False
        self.non_animated_background = None
        self.row = 0
        self.col = 10
        self.scanParms = {}
        self.ptychirecon_dir_selected = True
        
        # =============================================================================
        # XRF FOLDER SELECTION SECTION
        # =============================================================================
        # XRF Folder
        self.xrf_folder = tk.StringVar()
        self.xrf_folder.set(" ")
        xrffolder_button = tk.Button(
            self.setupfrm, text="XRF Folder", command=self.choose_xrf_folder
        )
        xrffolder_button.grid(column=0, row=0, padx=(5, 5), pady=(5, 5), sticky="W")
        xrf_folder_txt = tk.Label(self.setupfrm, textvariable=self.xrf_folder)
        xrf_folder_txt.grid(row=0, column=1, columnspan=4, sticky="W")

        # Scan Files
        scannum_txt = tk.Label(self.setupfrm, text="XRF Files:")
        scannum_txt.grid(column=0, row=1, pady=(5, 5), padx=(10, 0), sticky="W")
        
        self.scfile_sv = tk.StringVar()
        self.xrf_file_combobox = ttk.Combobox(
            self.setupfrm, textvariable=self.scfile_sv, state="readonly", width=11
        )
        self.xrf_file_combobox.grid(column=1, row=1, pady=(5, 5), sticky="W")
        self.xrf_file_combobox.bind("<<ComboboxSelected>>", self.load_xrf_scan)
        
        # Detector selection
        detector_sv = tk.StringVar()
        elm_txt = tk.Label(self.setupfrm, text="Elements:")
        elm_txt.grid(column=3, row=1, pady=(5, 5), sticky="W")
        self.detector_combobox = ttk.Combobox(
            self.setupfrm, textvariable=detector_sv, state="readonly", width=11
        )
        self.detector_combobox.grid(column=4, row=1, pady=(5, 5), sticky="W")
        self.detector_combobox.bind("<<ComboboxSelected>>", self.plot_data)

        loadscan_button = tk.Button(
            self.setupfrm, text="Update", command=self.updateFileList
        )
        loadscan_button.grid(column=4, row=0, padx=(5, 10), pady=(5, 5))
        
        # =============================================================================
        # PTYCHO FOLDER SELECTION SECTION
        # =============================================================================
        # Ptycho Folder
        self.ptycho_folder = tk.StringVar()
        self.ptycho_folder.set(" ")
        ptychofolder_button = tk.Button(
            self.setupfrm, text="Ptycho Folder", command=self.choose_ptycho_folder
        )
        ptychofolder_button.grid(column=0, row=2, padx=(5, 10), pady=(5, 5), sticky="W")
        ptycho_folder_txt = tk.Label(self.setupfrm, textvariable=self.ptycho_folder)
        ptycho_folder_txt.grid(row=2, column=1, columnspan=4, sticky="W")

        # Scan number
        ptycho_scannum_txt = tk.Label(self.setupfrm, text="Scan Number:")
        ptycho_scannum_txt.grid(column=0, row=3, pady=(5, 5), padx=(5, 0), sticky="W")
        
        self.ptycho_scannum_sv = tk.StringVar()
        self.ptycho_scannum_combobox = ttk.Combobox(
            self.setupfrm, textvariable=self.ptycho_scannum_sv, state="readonly", width=50
        )
        self.ptycho_scannum_combobox.grid(column=1, row=3, padx=(0, 0), pady=(5, 5), columnspan=4, sticky="W")
        self.ptycho_scannum_combobox.bind("<<ComboboxSelected>>", self.list_recon_directory)

        # ROI type
        ptycho_recon_txt = tk.Label(self.setupfrm, text="ROI Type (optional):")
        ptycho_recon_txt.grid(column=0, row=4, pady=(5, 5), padx=(5, 0), sticky="W")
        
        self.ptycho_roi_sv = tk.StringVar()
        self.ptycho_roi_combobox = ttk.Combobox(
            self.setupfrm, textvariable=self.ptycho_roi_sv, state="readonly", width=50
        )
        self.ptycho_roi_combobox.grid(column=1, row=4, padx=(0, 0), pady=(5, 5), sticky="W", columnspan=4)
        self.ptycho_roi_combobox.bind("<<ComboboxSelected>>", self.list_recon_directory)

        # Reconstruction method
        ptycho_recon_txt = tk.Label(self.setupfrm, text="Recon Method:")
        ptycho_recon_txt.grid(column=0, row=5, pady=(5, 5), padx=(5, 0), sticky="W")
        
        self.ptycho_recon_sv = tk.StringVar()
        self.ptycho_recon_combobox = ttk.Combobox(
            self.setupfrm, textvariable=self.ptycho_recon_sv, state="readonly", width=50
        )
        self.ptycho_recon_combobox.grid(column=1, row=5, padx=(0, 0), pady=(5, 5), sticky="W", columnspan=4)
        self.ptycho_recon_combobox.bind("<<ComboboxSelected>>", self.list_recon_directory)
        
        # Number of iterations
        ptycho_niter_txt = tk.Label(self.setupfrm, text="# Iterations:")
        ptycho_niter_txt.grid(column=0, row=6, pady=(5, 5), padx=(5, 0), sticky="W")
        
        self.ptycho_niter_sv = tk.StringVar()
        self.ptycho_niter_combobox = ttk.Combobox(
            self.setupfrm, textvariable=self.ptycho_niter_sv, state="readonly", width=50
        )
        self.ptycho_niter_combobox.grid(column=1, row=6, padx=(0, 0), pady=(5, 5), sticky="W", columnspan=4)
        self.ptycho_niter_combobox.bind("<<ComboboxSelected>>", self.load_ptycho_scan)

        # =============================================================================
        # MATPLOTLIB CANVAS SECTION
        # =============================================================================
        # Create matplotlib figure and canvas
        self.Figure2D = Figure()
        self.Axe2D = self.Figure2D.add_axes([0, 0, 1, 1])
        self.Axe2D.set_axis_off()
        
        self.Canvas2D = FigureCanvasTkAgg(self.Figure2D, master=self.setupfrm)
        self.Canvas2D.draw()
        self.Canvas2D.get_tk_widget().config(width=650, height=650, cursor="cross")
        self.Canvas2D.get_tk_widget().grid(
            column=0, columnspan=10, row=7, rowspan=50,
            padx=(10, 20), pady=(5, 5), sticky="W"
        )
        
        # Canvas event bindings
        self.Canvas2D_Mouse_Hover_Event = self.Canvas2D.mpl_connect(
            "motion_notify_event", self.Canvas2D_Mouser_Hover
        )
        self.Canvas2D_Button_Press_Event = self.Canvas2D.mpl_connect(
            "button_press_event", self.Canvas2D_Button_Pressed
        )
        self.Rectangle_Drawing = False
        self.Dot_Drawing = False
        self.non_animated_background = None

        # =============================================================================
        # CANVAS CONTROLS SECTION
        # =============================================================================
        # Log scale button
        self.log_button = tk.Button(
            self.setupfrm, text="Log", command=self.logscale_changed
        )
        self.log_button.grid(column=0, row=58, padx=(5, 5), pady=(0, 0))
        
        # Coordinate display
        self.xycorr = tk.StringVar()
        self.xycorr.set("x, y: (0.00, 0.00)")
        xycorr_label = tk.Label(self.setupfrm, textvariable=self.xycorr)
        xycorr_label.grid(row=58, column=1)
        
        # File status message
        self.openfilemsg = tk.StringVar()
        self.openfilemsg.set("")
        self.openfilemsg.trace("w", lambda *args: limit_stringvar_length(self.openfilemsg, 100))
        self.open_msg_label = tk.Label(self.setupfrm, textvariable=self.openfilemsg)
        self.open_msg_label.grid(row=59, column=0, columnspan=2)

        # =============================================================================
        # SCAN TYPE SELECTION SECTION
        # =============================================================================
        # Scan type label
        self.scantype_txt = tk.Label(
            self.setupfrm, text="1. Choose a type of scan below:"
        )
        self.scantype_txt.grid(
            row=self.row, column=self.col, padx=(0, 400), columnspan=10
        )
        
        # Scan type radio buttons
        self.scanType = tk.StringVar(self.setupfrm)
        self.scanType.set("XRF")
        scantype = ["XRF", "Coarse-Fine (Fixed Angle)", "Angle Sweep", "Coarse-Fine"]
        padx = [(18, 10), (0, 10), (18, 10), (0, 10)]
        # self.row = 2
        # self.col = 10
        self.row += 1
        
        for i, s in enumerate(scantype):
            scan_radio = ttk.Radiobutton(
                master=self.setupfrm, text=s, 
                variable=self.scanType, value=s
            )
            scan_radio.grid(
                row=self.row if i < 2 else (self.row+1), 
                column=self.col + i%2, padx=padx[i], stick='w'
            )
        
        # Ptycho checkbox
        self.ptychoVal = tk.IntVar()
        self.ptychoVal.set(0)
        ptycho_btn = ttk.Checkbutton(
            master=self.setupfrm, text='Ptycho Enabled', variable=self.ptychoVal
        )
        ptycho_btn.grid(row=self.row+1, column=self.col+2)
        
        self.row += 2

        # =============================================================================
        # INSERT METHOD SELECTION SECTION
        # =============================================================================
        # Insert method label
        self.inserttype_txt = tk.Label(
            self.setupfrm, text="2. Choose a method to insert scan parameters:"
        )
        self.inserttype_txt.grid(
            row=self.row, column=self.col, padx=(0, 320), columnspan=8
        )
        
        # Insert method radio buttons
        self.insertType = tk.StringVar(self.setupfrm)
        self.insertType.set("Manual")
        inserttype = ["Manual", "XY Center", "ScanBox"]
        padx = [(0, 84), (0, 120), (0, 120)]
        self.row += 1
        
        for i, s in enumerate(inserttype):
            insert_radio = ttk.Radiobutton(
                master=self.setupfrm, text=s, variable=self.insertType, value=s
            )
            insert_radio.grid(row=self.row, column=self.col + i, padx=padx[i])

        # =============================================================================
        # SCAN PARAMETERS INPUT SECTION
        # =============================================================================
        self.row += 2
        self.col = 10
        ncol = 3
        
        # Define input labels
        self.inputs_labels = [
            [
                "x_theta0", "y_theta0", "z_theta0", "x_scan", "y_scan", "z_scan",
                "dwell", "width", "w_step", "target_theta", "height", "h_step",
            ],
            [
                "theta_min", "theta_max", "theta_inc", "elm", "width_fine", "w_step_fine",
                "n_clusters", "height_fine", "h_step_fine", "sel_cluster", "dwell_fine",
                "use_mask", "mask_elm",
            ],
        ]
        
        # Create validation command
        temp_input = [0] * 6 + [20, 20, 1, 15, 10, 1]
        vcmd = self.setupfrm.register(checkEntryDigit)
        
        # Create basic scan parameter inputs
        for i, s in enumerate(self.inputs_labels[0]):
            if i % ncol == 0:
                self.row += 1
            sv = tk.StringVar()
            sv.trace("w", lambda name, index, mode, sv=sv: self.calcTime())
            
            # Label
            param_label = tk.Label(self.setupfrm, text=s + ":")
            param_label.grid(
                row=self.row, column=self.col + i % ncol, sticky="W", columnspan=2
            )
            
            # Entry field
            param_entry = tk.Entry(
                self.setupfrm, width=10, textvariable=sv,
                validate="all", validatecommand=(vcmd, "%P")
            )
            param_entry.grid(row=self.row, column=self.col + i % ncol, padx=(70, 0))
            param_entry.insert(0, str(temp_input[i]))
            self.scanParms.update({s: param_entry})

        # =============================================================================
        # SAMPLE AND BDA SECTION
        # =============================================================================
        self.row += 1
        
        # Sample name
        smp_txt = tk.Label(self.setupfrm, text="Sample name:")
        smp_txt.grid(row=self.row, column=self.col, sticky="w")
        
        self.smp_name = tk.StringVar()
        smp_entry = tk.Entry(self.setupfrm, textvariable=self.smp_name)
        smp_entry.grid(
            row=self.row, column=self.col, columnspan=2, sticky="w", padx=(88, 0)
        )
        
        # BDA Position
        bda_txt = tk.Label(self.setupfrm, text="BDA Position:")
        bda_txt.grid(row=self.row, column=self.col + 1, sticky="w", padx=(100, 0))
        
        self.bda = tk.StringVar()
        self.bda.set("%.2f" % (self.pvComm.getBDAx()))
        bda_entry = tk.Entry(
            self.setupfrm, textvariable=self.bda,
            validate="all", validatecommand=(vcmd, "%P")
        )
        bda_entry.grid(
            row=self.row, column=self.col + 2, sticky="w", padx=(25, 0), columnspan=2
        )

        # =============================================================================
        # XYZ UPDATE BUTTONS SECTION
        # =============================================================================
        self.row += 1
        
        # XYZ update buttons
        updatexyz0_btn = tk.Button(
            self.setupfrm, text="XYZ_0 + transform", command=self.updatexyz0
        )
        updatexyz0_btn.grid(row=self.row, column=self.col, columnspan=2)
        
        updatexyz_btn = tk.Button(
            self.setupfrm, text="XYZ from PV", command=self.updatexyz
        )
        updatexyz_btn.grid(row=self.row, column=self.col + 1, columnspan=2)

        # =============================================================================
        # TIME CALCULATION SECTION
        # =============================================================================
        self.row += 1
        
        # Estimated scan time
        calctime_txt = tk.Label(self.setupfrm, text="Estimated scan time (min):")
        calctime_txt.grid(row=self.row, column=self.col, sticky="w")
        
        self.calctime = tk.StringVar()
        self.calctime.set("0")
        self.calctime_out = tk.Label(
            self.setupfrm, width=10, textvariable=self.calctime
        )
        self.calctime_out.grid(row=self.row, column=self.col + 1, sticky="w")
        
        # XYZ update message
        self.row += 1
        self.updatexyz_msg = tk.StringVar()
        self.updatexyz_msg.set("")
        self.updatexyz_label = tk.Label(self.setupfrm, textvariable=self.updatexyz_msg)
        self.updatexyz_label.grid(
            row=self.row, column=self.col, sticky="w", columnspan=3
        )

        # =============================================================================
        # TOMOGRAPHY PARAMETERS SECTION
        # =============================================================================
        self.row += 4
        
        # Tomography parameters label
        tomography_txt = tk.Label(
            self.setupfrm, text="Additional scan parameters for tomography"
        )
        tomography_txt.grid(
            row=self.row, column=self.col, padx=(0, 0), columnspan=8
        )
        
        # Create tomography parameter inputs
        for i, s in enumerate(self.inputs_labels[1]):
            if i % ncol == 0:
                self.row += 1
                    
            # Label
            tom_label = tk.Label(self.setupfrm, text=s + ":")
            tom_label.grid(
                row=self.row, column=self.col + i % ncol, sticky="W", columnspan=2
            )
            
            # Entry field
            if (s == "elm") | (s == "mask_elm"):
                tom_entry = tk.Entry(self.setupfrm, width=10)
                tom_entry.insert(0, "P")
            else:
                tom_entry = tk.Entry(
                    self.setupfrm, width=10, validate="all", validatecommand=(vcmd, "%P")
                )
                tom_entry.insert(0, "0")
            tom_entry.grid(row=self.row, column=self.col + i % ncol, padx=(70, 0))
            self.scanParms.update({s: tom_entry})

        # =============================================================================
        # DIRECTORY AND TIME DISPLAY SECTION
        # =============================================================================
        row = self.row + 7
        
        # Current data save directory
        self.pardir = tk.StringVar()
        self.pardir.set(self.pvComm.getDir())
        scdir_txt = tk.Button(
            self.setupfrm, text="Current data save directory:", command=self.getUserDir
        )
        scdir_txt.grid(row=row, column=self.col, sticky="w")
        
        pardir_out = tk.Label(self.setupfrm, textvariable=self.pardir)
        pardir_out.grid(row=row, column=self.col + 1, sticky="w", columnspan=3)

        # Estimated total scan time
        row += 2
        tot_est_txt = tk.Label(self.setupfrm, text="Estimated total scan time:")
        tot_est_txt.grid(row=row, column=self.col, sticky="w")
        
        self.tot_time = tk.StringVar()
        self.tot_time.set("0")
        tot_est_val = tk.Label(self.setupfrm, textvariable=self.tot_time)
        tot_est_val.grid(row=row, column=self.col + 1, sticky="w")
