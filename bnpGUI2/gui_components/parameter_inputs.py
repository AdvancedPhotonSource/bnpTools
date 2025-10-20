"""
Parameter input GUI components
Extracted from original setupFrame.py
"""

import tkinter as tk
from tkinter import ttk
from typing import Dict, List, Callable, Optional
from ..utils.validation_utils import checkEntryDigit


class ParameterInputWidget:
    """Widget for scan parameter inputs"""
    
    def __init__(self, parent_frame, pv_comm):
        self.parent_frame = parent_frame
        self.pv_comm = pv_comm
        self.scan_parms = {}
        
        # Variables
        self.scan_type = tk.StringVar(parent_frame)
        self.scan_type.set("XRF")
        self.insert_type = tk.StringVar(parent_frame)
        self.insert_type.set("Manual")
        self.ptycho_val = tk.IntVar()
        self.ptycho_val.set(0)
        self.smp_name = tk.StringVar()
        self.bda = tk.StringVar()
        self.bda.set("%.2f" % (self.pv_comm.getBDAx()))
        self.calctime = tk.StringVar()
        self.calctime.set("0")
        self.updatexyz_msg = tk.StringVar()
        self.updatexyz_msg.set("")
        
        # Callbacks
        self.calc_time_callback = None
        self.update_xyz_callback = None
        
        self._create_widgets()
    
    def _create_widgets(self):
        """Create parameter input widgets"""
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
        vcmd = self.parent_frame.register(checkEntryDigit)
        
        # Scan type selection
        self._create_scan_type_section()
        
        # Insert method selection
        self._create_insert_method_section()
        
        # Basic scan parameters
        self._create_basic_parameters(vcmd)
        
        # Sample and BDA
        self._create_sample_bda_section(vcmd)
        
        # XYZ update buttons
        self._create_xyz_update_section()
        
        # Time calculation
        self._create_time_calculation_section()
        
        # Tomography parameters
        self._create_tomography_parameters(vcmd)
    
    def _create_scan_type_section(self):
        """Create scan type selection section"""
        row = 0
        col = 10
        
        # Scan type label
        scantype_txt = tk.Label(
            self.parent_frame, text="1. Choose a type of scan below:"
        )
        scantype_txt.grid(
            row=row, column=col, padx=(0, 400), columnspan=10
        )
        
        # Scan type radio buttons
        scantype = ["XRF", "Coarse-Fine (Fixed Angle)", "Angle Sweep", "Coarse-Fine"]
        padx = [(18, 10), (0, 10), (18, 10), (0, 10)]
        row += 1
        
        for i, s in enumerate(scantype):
            scan_radio = ttk.Radiobutton(
                master=self.parent_frame, text=s, 
                variable=self.scan_type, value=s
            )
            scan_radio.grid(
                row=row if i < 2 else (row+1), 
                column=col + i%2, padx=padx[i], stick='w'
            )
        
        # Ptycho checkbox
        ptycho_btn = ttk.Checkbutton(
            master=self.parent_frame, text='Ptycho Enabled', variable=self.ptycho_val
        )
        ptycho_btn.grid(row=row+1, column=col+2)
    
    def _create_insert_method_section(self):
        """Create insert method selection section"""
        row = 2
        col = 10
        
        # Insert method label
        inserttype_txt = tk.Label(
            self.parent_frame, text="2. Choose a method to insert scan parameters:"
        )
        inserttype_txt.grid(
            row=row, column=col, padx=(0, 320), columnspan=8
        )
        
        # Insert method radio buttons
        inserttype = ["Manual", "XY Center", "ScanBox"]
        padx = [(0, 84), (0, 120), (0, 120)]
        row += 1
        
        for i, s in enumerate(inserttype):
            insert_radio = ttk.Radiobutton(
                master=self.parent_frame, text=s, variable=self.insert_type, value=s
            )
            insert_radio.grid(row=row, column=col + i, padx=padx[i])
    
    def _create_basic_parameters(self, vcmd):
        """Create basic scan parameter inputs"""
        row = 4
        col = 10
        ncol = 3
        
        # Create basic scan parameter inputs
        for i, s in enumerate(self.inputs_labels[0]):
            if i % ncol == 0:
                row += 1
            sv = tk.StringVar()
            sv.trace("w", lambda name, index, mode, sv=sv: self._on_parameter_changed())
            
            # Label
            param_label = tk.Label(self.parent_frame, text=s + ":")
            param_label.grid(
                row=row, column=col + i % ncol, sticky="W", columnspan=2
            )
            
            # Entry field
            param_entry = tk.Entry(
                self.parent_frame, width=10, textvariable=sv,
                validate="all", validatecommand=(vcmd, "%P")
            )
            param_entry.grid(row=row, column=col + i % ncol, padx=(70, 0))
            
            # Set default values
            temp_input = [0] * 6 + [20, 20, 1, 15, 10, 1]
            param_entry.insert(0, str(temp_input[i]))
            self.scan_parms.update({s: param_entry})
    
    def _create_sample_bda_section(self, vcmd):
        """Create sample name and BDA position section"""
        row = 8
        
        # Sample name
        smp_txt = tk.Label(self.parent_frame, text="Sample name:")
        smp_txt.grid(row=row, column=10, sticky="w")
        
        smp_entry = tk.Entry(self.parent_frame, textvariable=self.smp_name)
        smp_entry.grid(
            row=row, column=10, columnspan=2, sticky="w", padx=(88, 0)
        )
        
        # BDA Position
        bda_txt = tk.Label(self.parent_frame, text="BDA Position:")
        bda_txt.grid(row=row, column=11, sticky="w", padx=(100, 0))
        
        bda_entry = tk.Entry(
            self.parent_frame, textvariable=self.bda,
            validate="all", validatecommand=(vcmd, "%P")
        )
        bda_entry.grid(
            row=row, column=12, sticky="w", padx=(25, 0), columnspan=2
        )
    
    def _create_xyz_update_section(self):
        """Create XYZ update buttons section"""
        row = 9
        
        # XYZ update buttons
        updatexyz0_btn = tk.Button(
            self.parent_frame, text="XYZ_0 + transform", command=self.updatexyz0
        )
        updatexyz0_btn.grid(row=row, column=10, columnspan=2)
        
        updatexyz_btn = tk.Button(
            self.parent_frame, text="XYZ from PV", command=self.updatexyz
        )
        updatexyz_btn.grid(row=row, column=11, columnspan=2)
    
    def _create_time_calculation_section(self):
        """Create time calculation section"""
        row = 10
        
        # Estimated scan time
        calctime_txt = tk.Label(self.parent_frame, text="Estimated scan time (min):")
        calctime_txt.grid(row=row, column=10, sticky="w")
        
        self.calctime_out = tk.Label(
            self.parent_frame, width=10, textvariable=self.calctime
        )
        self.calctime_out.grid(row=row, column=11, sticky="w")
        
        # XYZ update message
        row += 1
        self.updatexyz_label = tk.Label(self.parent_frame, textvariable=self.updatexyz_msg)
        self.updatexyz_label.grid(
            row=row, column=10, sticky="w", columnspan=3
        )
    
    def _create_tomography_parameters(self, vcmd):
        """Create tomography parameters section"""
        row = 12
        col = 10
        ncol = 3
        
        # Tomography parameters label
        tomography_txt = tk.Label(
            self.parent_frame, text="Additional scan parameters for tomography"
        )
        tomography_txt.grid(
            row=row, column=col, padx=(0, 0), columnspan=8
        )
        
        # Create tomography parameter inputs
        for i, s in enumerate(self.inputs_labels[1]):
            if i % ncol == 0:
                row += 1
                    
            # Label
            tom_label = tk.Label(self.parent_frame, text=s + ":")
            tom_label.grid(
                row=row, column=col + i % ncol, sticky="W", columnspan=2
            )
            
            # Entry field
            if (s == "elm") | (s == "mask_elm"):
                tom_entry = tk.Entry(self.parent_frame, width=10)
                tom_entry.insert(0, "P")
            else:
                tom_entry = tk.Entry(
                    self.parent_frame, width=10, validate="all", validatecommand=(vcmd, "%P")
                )
                tom_entry.insert(0, "0")
            tom_entry.grid(row=row, column=col + i % ncol, padx=(70, 0))
            self.scan_parms.update({s: tom_entry})
    
    def _on_parameter_changed(self):
        """Handle parameter change"""
        if self.calc_time_callback:
            self.calc_time_callback()
    
    def calc_time(self):
        """Calculate estimated scan time"""
        strlist = ["width", "height", "w_step", "h_step", "dwell"]
        try:
            value = [float(self.scan_parms[s].get()) for s in strlist]
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
    
    def updatexyz0(self):
        """Update XYZ0 and transform coordinates"""
        if (self.pv_comm.getSMAngle() ** 2) < 1e-3:
            self.fillxyz0(empty=False)
            self.coordtform()
            self.updatexyz_msg.set(
                "x-, y-, z- theta0 updated\n"
                + "Coordinate transformed. x-, y-, z- scan values updated"
            )
            self.updatexyz_label.config(fg="green")
        else:
            self.updatexyz_msg.set("Theta (Rotation) motor is not at 0")
            self.updatexyz_label.config(fg="red")
    
    def updatexyz(self):
        """Update XYZ from PV"""
        if (self.pv_comm.getSMAngle() ** 2) < 1e-3:
            self.fillxyz0(empty=False)
            self.fillxyzScan()
        else:
            self.fillxyz0()
            self.fillxyzScan(empty=False)
    
    def fillxyz0(self, empty=True):
        """Fill XYZ0 values"""
        if empty:
            x, y, z = ['', '', '']
        else:
            x, y, z = self.pv_comm.getXYZcenter()
        for a_, v_ in zip(["x_theta0", "y_theta0", "z_theta0"], [x, y, z]):
            self.scan_parms[a_].delete(0, tk.END)
            self.scan_parms[a_].insert(0, str(v_))
    
    def fillxyzScan(self, empty=True):
        """Fill XYZ scan values"""
        if empty:
            x, y, z, theta = ['', '', '', '']
        else:
            x, y, z = self.pv_comm.getXYZcenter()
            theta = self.pv_comm.getSMAngle()
        for a_, v_ in zip(['x_scan', 'y_scan', 'z_scan', 'target_theta'], [x, y, z, theta]):
            self.scan_parms[a_].delete(0, tk.END)
            self.scan_parms[a_].insert(0, str(v_))
    
    def coordtform(self):
        """Coordinate transformation"""
        from ..utils.coordinate_utils import coordinate_transform
        
        fields = ["target_theta", "x_theta0", "y_theta0", "z_theta0"]
        fvals = [float(self.scan_parms[a_].get()) for a_ in fields]
        ctform = coordinate_transform(*fvals)
        writeFields = ["x_scan", "y_scan", "z_scan"]
        wfv = [round(ctform[a], 2) for a in ["x", "y", "z"]]
        for a_, v_ in zip(writeFields, wfv):
            self.scan_parms[a_].delete(0, tk.END)
            self.scan_parms[a_].insert(0, str(v_))
        self.updatexyz_msg.set("Coordinate transformed. x-, y-, z- scan values updated")
        self.updatexyz_label.config(fg="green")
    
    def set_calc_time_callback(self, callback: Callable):
        """Set callback for time calculation"""
        self.calc_time_callback = callback
