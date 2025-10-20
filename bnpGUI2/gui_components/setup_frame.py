"""
Main setup frame that orchestrates all GUI components
Extracted from original setupFrame.py
"""

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional, Dict, Any

from .file_selection import FileSelectionWidget
from .parameter_inputs import ParameterInputWidget
from .image_canvas import ImageCanvasWidget
from .scan_controls import ScanControlsWidget
from ..data_handlers.xrf_handler import XRFDataHandler
from ..data_handlers.ptycho_handler import PtychoDataHandler
from ..utils.validation_utils import limit_stringvar_length


class SetupFrame:
    """Main setup frame that coordinates all GUI components"""
    
    def __init__(self, tab_control, pv_comm):
        self.tab_control = tab_control
        self.pv_comm = pv_comm
        
        # Create main frame
        self.setupfrm = ttk.Frame(tab_control)
        
        # Initialize data handlers
        self.xrf_handler = XRFDataHandler(pv_comm)
        self.ptycho_handler = PtychoDataHandler(pv_comm)
        
        # Initialize GUI components
        self.file_selection = FileSelectionWidget(
            self.setupfrm, pv_comm, self.xrf_handler, self.ptycho_handler
        )
        self.parameter_inputs = ParameterInputWidget(self.setupfrm, pv_comm)
        self.image_canvas = ImageCanvasWidget(self.setupfrm)
        self.scan_controls = ScanControlsWidget(self.setupfrm, pv_comm)
        
        # Set up callbacks
        self._setup_callbacks()
        
        # Create add to scan button
        self.scan_controls.create_add_scan_button(0, 10)
    
    def _setup_callbacks(self):
        """Set up callbacks between components"""
        # File selection callbacks
        self.file_selection.set_xrf_load_callback(self._load_xrf_scan)
        self.file_selection.set_ptycho_load_callback(self._load_ptycho_scan)
        
        # Parameter input callbacks
        self.parameter_inputs.set_calc_time_callback(self._calc_time)
        
        # Image canvas callbacks
        self.image_canvas.set_xy_center_callback(self._update_xy_center)
        self.image_canvas.set_scan_box_callback(self._update_scan_box)
        
        # Scan controls callbacks
        self.scan_controls.set_add_scan_callback(self._add_to_scan)
    
    def _load_xrf_scan(self):
        """Load XRF scan data"""
        folder = self.file_selection.xrf_folder.get()
        filename = self.file_selection.xrf_file_combobox.get()
        
        if not folder or not filename:
            return
        
        result = self.xrf_handler.load_xrf_scan(folder, filename)
        
        if "error" in result:
            self.scan_controls.set_file_status(result["error"], "red")
            return
        
        # Update detector combobox
        self.file_selection.detector_combobox["values"] = result["detectors"]
        if result["detectors"]:
            self.file_selection.detector_combobox.current(0)
        
        # Update image canvas
        detector_index = self.file_selection.detector_combobox.current()
        plot_data = self.xrf_handler.get_plot_data(detector_index)
        
        if plot_data is not None:
            self.image_canvas.display_image(
                plot_data, 
                x=result["x"], 
                y=result["y"], 
                cmap="inferno"
            )
            
            # Update file info
            self.file_selection.file_z = result["file_z"]
            self.file_selection.file_theta = result["file_theta"]
            
            if result["file_z"] is None:
                self.scan_controls.set_file_status(
                    f"Samz PV not found {self.xrf_handler.h5_filename}", "red"
                )
            else:
                self.scan_controls.set_file_status(
                    f"{filename} is open", "green"
                )
    
    def _load_ptycho_scan(self):
        """Load Ptycho scan data"""
        folder = self.file_selection.ptycho_folder.get()
        scan_num = self.file_selection.ptycho_scannum_combobox.get()
        roi_folder = self.file_selection.ptycho_roi_combobox.get()
        recon_method = self.file_selection.ptycho_recon_combobox.get()
        niter_file = self.file_selection.ptycho_niter_combobox.get()
        
        if not all([folder, scan_num, roi_folder, recon_method, niter_file]):
            return
        
        result = self.ptycho_handler.load_ptycho_scan(
            folder, scan_num, roi_folder, recon_method, niter_file
        )
        
        if "error" in result:
            self.scan_controls.set_file_status(result["error"], "red")
            return
        
        # Update image canvas
        self.image_canvas.display_image(
            result["tiff"],
            x=result["x"],
            y=result["y"],
            cmap="gray"
        )
        
        # Update file info
        self.file_selection.file_z = result["file_z"]
        self.file_selection.file_theta = result["file_theta"]
        
        if result["file_z"] is None:
            self.scan_controls.set_file_status(
                f"Samz PV not found {self.ptycho_handler.mda_filepath}", "red"
            )
        else:
            msg = f"{scan_num} - {roi_folder} - {niter_file} is open"
            self.scan_controls.set_file_status(msg, "green")
    
    def _calc_time(self):
        """Calculate scan time"""
        self.parameter_inputs.calc_time()
    
    def _update_xy_center(self, xstart, ystart):
        """Update XY center from canvas click"""
        if (self.image_canvas.x is None or self.image_canvas.y is None or 
            self.file_selection.file_z is None or self.file_selection.file_theta is None):
            return
        
        x_scan = round(self.image_canvas.x[xstart], 2)
        y_scan = round(self.image_canvas.y[ystart], 2)
        z_scan = round(self.file_selection.file_z, 2)
        target_theta = self.file_selection.file_theta
        
        # Update parameter inputs
        slabel = ["x_scan", "y_scan", "z_scan", "target_theta"]
        values = [x_scan, y_scan, z_scan, target_theta]
        
        for s, v in zip(slabel, values):
            self.parameter_inputs.scan_parms[s].delete(0, tk.END)
            self.parameter_inputs.scan_parms[s].insert(0, f"{v:.2f}")
        
        # Update theta0 values
        slabel = ["x_theta0", "y_theta0", "z_theta0"]
        if (target_theta ** 2) < 1e-3:
            svlabel = ["x_scan", "y_scan", "z_scan"]
        else:
            svlabel = ["", "", ""]  # Empty values
        
        for s_, sv_ in zip(slabel, svlabel):
            self.parameter_inputs.scan_parms[s_].delete(0, tk.END)
            if sv_:
                value = self.parameter_inputs.scan_parms[sv_].get()
                self.parameter_inputs.scan_parms[s_].insert(0, value)
            else:
                self.parameter_inputs.scan_parms[s_].insert(0, "")
    
    def _update_scan_box(self, xmin, xmax, ymin, ymax):
        """Update scan box parameters from canvas selection"""
        if (self.image_canvas.x is None or self.image_canvas.y is None):
            return
        
        x_scan = round((self.image_canvas.x[xmin] + self.image_canvas.x[xmax]) / 2, 2)
        y_scan = round((self.image_canvas.y[ymin] + self.image_canvas.y[ymax]) / 2, 2)
        width = abs(self.image_canvas.x[xmin] - self.image_canvas.x[xmax])
        height = abs(self.image_canvas.y[ymin] - self.image_canvas.y[ymax])
        
        if self.file_selection.file_z is not None:
            z_scan = round(self.file_selection.file_z, 2)
            target_theta = self.file_selection.file_theta
            
            slabel = ["x_scan", "y_scan", "width", "height", "z_scan", "target_theta"]
            values = [x_scan, y_scan, width, height, z_scan, target_theta]
            
            for s, v in zip(slabel, values):
                self.parameter_inputs.scan_parms[s].delete(0, tk.END)
                self.parameter_inputs.scan_parms[s].insert(0, f"{v:.2f}")
            
            # Update theta0 values
            slabel = ["x_theta0", "y_theta0", "z_theta0"]
            if (target_theta ** 2) < 1e-3:
                svlabel = ["x_scan", "y_scan", "z_scan"]
            else:
                svlabel = ["", "", ""]
            
            for s_, sv_ in zip(slabel, svlabel):
                self.parameter_inputs.scan_parms[s_].delete(0, tk.END)
                if sv_:
                    value = self.parameter_inputs.scan_parms[sv_].get()
                    self.parameter_inputs.scan_parms[s_].insert(0, value)
                else:
                    self.parameter_inputs.scan_parms[s_].insert(0, "")
        else:
            slabel = ["x_scan", "y_scan", "width", "height"]
            values = [x_scan, y_scan, width, height]
            
            for s, v in zip(slabel, values):
                self.parameter_inputs.scan_parms[s].delete(0, tk.END)
                self.parameter_inputs.scan_parms[s].insert(0, f"{v:.2f}")
            
            # Clear theta values
            flabel = ['z_scan', 'target_theta', 'x_theta0', 'y_theta0', 'z_theta0']
            for f in flabel:
                self.parameter_inputs.scan_parms[f].delete(0, tk.END)
    
    def _add_to_scan(self):
        """Add current parameters to scan list"""
        # This would be implemented to add scan parameters to a scan queue
        # For now, just print the parameters
        print("Adding scan with parameters:")
        for key, widget in self.parameter_inputs.scan_parms.items():
            print(f"  {key}: {widget.get()}")
    
    def add_to_scan_btn(self, func: Callable):
        """Add to scan button callback (for compatibility with original)"""
        self.scan_controls.set_add_scan_callback(func)
    
    def get_scan_parameters(self) -> Dict[str, Any]:
        """Get current scan parameters"""
        params = {}
        for key, widget in self.parameter_inputs.scan_parms.items():
            params[key] = widget.get()
        
        # Add other parameters
        params.update({
            'scan_type': self.parameter_inputs.scan_type.get(),
            'insert_type': self.parameter_inputs.insert_type.get(),
            'ptycho_enabled': bool(self.parameter_inputs.ptycho_val.get()),
            'sample_name': self.parameter_inputs.smp_name.get(),
            'bda_position': self.parameter_inputs.bda.get(),
        })
        
        return params
