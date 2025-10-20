"""
File selection GUI components
Extracted from original setupFrame.py
"""

import tkinter as tk
from tkinter import ttk, filedialog
from typing import Callable, Dict, List, Optional


class FileSelectionWidget:
    """Widget for file and folder selection"""
    
    def __init__(self, parent_frame, pv_comm, xrf_handler, ptycho_handler):
        self.parent_frame = parent_frame
        self.pv_comm = pv_comm
        self.xrf_handler = xrf_handler
        self.ptycho_handler = ptycho_handler
        
        # Variables
        self.xrf_folder = tk.StringVar()
        self.xrf_folder.set(" ")
        self.ptycho_folder = tk.StringVar()
        self.ptycho_folder.set(" ")
        
        # Callbacks
        self.xrf_load_callback = None
        self.ptycho_load_callback = None
        
        self._create_widgets()
    
    def _create_widgets(self):
        """Create file selection widgets"""
        # XRF Folder Selection
        xrf_folder_button = tk.Button(
            self.parent_frame, text="XRF Folder", command=self.choose_xrf_folder
        )
        xrf_folder_button.grid(column=0, row=0, padx=(5, 5), pady=(5, 5), sticky="W")
        
        xrf_folder_txt = tk.Label(self.parent_frame, textvariable=self.xrf_folder)
        xrf_folder_txt.grid(row=0, column=1, columnspan=4, sticky="W")
        
        # XRF Files
        scannum_txt = tk.Label(self.parent_frame, text="XRF Files:")
        scannum_txt.grid(column=0, row=1, pady=(5, 5), padx=(10, 0), sticky="W")
        
        self.scfile_sv = tk.StringVar()
        self.xrf_file_combobox = ttk.Combobox(
            self.parent_frame, textvariable=self.scfile_sv, state="readonly", width=11
        )
        self.xrf_file_combobox.grid(column=1, row=1, pady=(5, 5), sticky="W")
        self.xrf_file_combobox.bind("<<ComboboxSelected>>", self._on_xrf_file_selected)
        
        # Detector selection
        detector_sv = tk.StringVar()
        elm_txt = tk.Label(self.parent_frame, text="Elements:")
        elm_txt.grid(column=3, row=1, pady=(5, 5), sticky="W")
        self.detector_combobox = ttk.Combobox(
            self.parent_frame, textvariable=detector_sv, state="readonly", width=11
        )
        self.detector_combobox.grid(column=4, row=1, pady=(5, 5), sticky="W")
        self.detector_combobox.bind("<<ComboboxSelected>>", self._on_detector_selected)
        
        loadscan_button = tk.Button(
            self.parent_frame, text="Update", command=self.update_file_list
        )
        loadscan_button.grid(column=4, row=0, padx=(5, 10), pady=(5, 5))
        
        # Ptycho Folder Selection
        ptychofolder_button = tk.Button(
            self.parent_frame, text="Ptycho Folder", command=self.choose_ptycho_folder
        )
        ptychofolder_button.grid(column=0, row=2, padx=(5, 10), pady=(5, 5), sticky="W")
        
        ptycho_folder_txt = tk.Label(self.parent_frame, textvariable=self.ptycho_folder)
        ptycho_folder_txt.grid(row=2, column=1, columnspan=4, sticky="W")
        
        # Scan number
        ptycho_scannum_txt = tk.Label(self.parent_frame, text="Scan Number:")
        ptycho_scannum_txt.grid(column=0, row=3, pady=(5, 5), padx=(5, 0), sticky="W")
        
        self.ptycho_scannum_sv = tk.StringVar()
        self.ptycho_scannum_combobox = ttk.Combobox(
            self.parent_frame, textvariable=self.ptycho_scannum_sv, state="readonly", width=50
        )
        self.ptycho_scannum_combobox.grid(column=1, row=3, padx=(0, 0), pady=(5, 5), 
                                        columnspan=4, sticky="W")
        self.ptycho_scannum_combobox.bind("<<ComboboxSelected>>", self._on_ptycho_scan_selected)
        
        # ROI type
        ptycho_recon_txt = tk.Label(self.parent_frame, text="ROI Type (optional):")
        ptycho_recon_txt.grid(column=0, row=4, pady=(5, 5), padx=(5, 0), sticky="W")
        
        self.ptycho_roi_sv = tk.StringVar()
        self.ptycho_roi_combobox = ttk.Combobox(
            self.parent_frame, textvariable=self.ptycho_roi_sv, state="readonly", width=50
        )
        self.ptycho_roi_combobox.grid(column=1, row=4, padx=(0, 0), pady=(5, 5), 
                                    sticky="W", columnspan=4)
        self.ptycho_roi_combobox.bind("<<ComboboxSelected>>", self._on_ptycho_roi_selected)
        
        # Reconstruction method
        ptycho_recon_txt = tk.Label(self.parent_frame, text="Recon Method:")
        ptycho_recon_txt.grid(column=0, row=5, pady=(5, 5), padx=(5, 0), sticky="W")
        
        self.ptycho_recon_sv = tk.StringVar()
        self.ptycho_recon_combobox = ttk.Combobox(
            self.parent_frame, textvariable=self.ptycho_recon_sv, state="readonly", width=50
        )
        self.ptycho_recon_combobox.grid(column=1, row=5, padx=(0, 0), pady=(5, 5), 
                                      sticky="W", columnspan=4)
        self.ptycho_recon_combobox.bind("<<ComboboxSelected>>", self._on_ptycho_recon_selected)
        
        # Number of iterations
        ptycho_niter_txt = tk.Label(self.parent_frame, text="# Iterations:")
        ptycho_niter_txt.grid(column=0, row=6, pady=(5, 5), padx=(5, 0), sticky="W")
        
        self.ptycho_niter_sv = tk.StringVar()
        self.ptycho_niter_combobox = ttk.Combobox(
            self.parent_frame, textvariable=self.ptycho_niter_sv, state="readonly", width=50
        )
        self.ptycho_niter_combobox.grid(column=1, row=6, padx=(0, 0), pady=(5, 5), 
                                      sticky="W", columnspan=4)
        self.ptycho_niter_combobox.bind("<<ComboboxSelected>>", self._on_ptycho_niter_selected)
    
    def choose_xrf_folder(self):
        """Choose XRF folder"""
        initial_dir = self.xrf_handler.choose_xrf_folder()
        h5_folder = filedialog.askdirectory(initialdir=initial_dir)
        
        if h5_folder:
            self.xrf_folder.set(h5_folder)
            if len(h5_folder) > 10:
                files = self.xrf_handler.get_h5_files(h5_folder)
                self.xrf_file_combobox["values"] = files
            else:
                h5_folder = filedialog.askdirectory(initialdir="/mnt/micdata1/bnp/2021-3/")
    
    def choose_ptycho_folder(self):
        """Choose Ptycho folder"""
        ptycho_dir = self.ptycho_handler.choose_ptycho_folder(self.ptycho_folder.get())
        ptycho_folder = filedialog.askdirectory(initialdir=ptycho_dir)
        
        if len(ptycho_folder) > 10:
            self.ptycho_folder.set(ptycho_folder)
            scan_numbers = self.ptycho_handler.get_scan_numbers(ptycho_folder)
            self.ptycho_scannum_combobox["values"] = scan_numbers
            
            # Clear dependent comboboxes
            self.ptycho_scannum_combobox.set("")
            self.ptycho_recon_combobox.set("")
            self.ptycho_roi_combobox.set("")
            self.ptycho_niter_combobox.set("")
        else:
            self.ptycho_folder.set(ptycho_dir)
    
    def update_file_list(self):
        """Update XRF file list"""
        files = self.xrf_handler.get_h5_files(self.xrf_folder.get())
        self.xrf_file_combobox["values"] = files
    
    def _on_xrf_file_selected(self, event):
        """Handle XRF file selection"""
        if self.xrf_load_callback:
            self.xrf_load_callback()
    
    def _on_detector_selected(self, event):
        """Handle detector selection"""
        if self.xrf_load_callback:
            self.xrf_load_callback()
    
    def _on_ptycho_scan_selected(self, event):
        """Handle ptycho scan selection"""
        self._update_ptycho_comboboxes()
        if self.ptycho_load_callback:
            self.ptycho_load_callback()
    
    def _on_ptycho_roi_selected(self, event):
        """Handle ptycho ROI selection"""
        self._update_ptycho_comboboxes()
        if self.ptycho_load_callback:
            self.ptycho_load_callback()
    
    def _on_ptycho_recon_selected(self, event):
        """Handle ptycho reconstruction method selection"""
        self._update_ptycho_comboboxes()
        if self.ptycho_load_callback:
            self.ptycho_load_callback()
    
    def _on_ptycho_niter_selected(self, event):
        """Handle ptycho iteration selection"""
        if self.ptycho_load_callback:
            self.ptycho_load_callback()
    
    def _update_ptycho_comboboxes(self):
        """Update ptycho comboboxes based on selections"""
        # Clear dependent comboboxes
        self.ptycho_roi_combobox["values"] = []
        self.ptycho_recon_combobox["values"] = []
        self.ptycho_niter_combobox["values"] = []
        self.ptycho_roi_combobox.set("")
        self.ptycho_recon_combobox.set("")
        self.ptycho_niter_combobox.set("")
        
        if not self.ptycho_scannum_combobox.get():
            return
        
        # Get reconstruction directory info
        result = self.ptycho_handler.list_recon_directory(
            self.ptycho_folder.get(), 
            self.ptycho_scannum_combobox.get()
        )
        
        if "error" in result:
            return
        
        # Update comboboxes
        self.ptycho_roi_combobox["values"] = result["roi_folders"]
        self.ptycho_recon_combobox["values"] = result["recon_methods"]
        
        if result["roi_folders"]:
            self.ptycho_roi_combobox.current(0)
        if result["recon_methods"]:
            self.ptycho_recon_combobox.current(0)
        
        # Update iteration list
        if (self.ptycho_roi_combobox.get() and 
            self.ptycho_recon_combobox.get()):
            niter_list = self.ptycho_handler.get_niter_list(
                self.ptycho_folder.get(),
                self.ptycho_scannum_combobox.get(),
                self.ptycho_roi_combobox.get(),
                self.ptycho_recon_combobox.get()
            )
            self.ptycho_niter_combobox["values"] = niter_list
            if niter_list:
                self.ptycho_niter_combobox.current(0)
    
    def set_xrf_load_callback(self, callback: Callable):
        """Set callback for XRF data loading"""
        self.xrf_load_callback = callback
    
    def set_ptycho_load_callback(self, callback: Callable):
        """Set callback for Ptycho data loading"""
        self.ptycho_load_callback = callback
