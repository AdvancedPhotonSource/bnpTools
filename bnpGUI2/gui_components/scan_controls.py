"""
Scan controls GUI component
Extracted from original setupFrame.py
"""

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional


class ScanControlsWidget:
    """Widget for scan control buttons and status display"""
    
    def __init__(self, parent_frame, pv_comm):
        self.parent_frame = parent_frame
        self.pv_comm = pv_comm
        
        # Variables
        self.pardir = tk.StringVar()
        self.pardir.set(self.pv_comm.getDir())
        self.tot_time = tk.StringVar()
        self.tot_time.set("0")
        self.openfilemsg = tk.StringVar()
        self.openfilemsg.set("")
        
        # Callbacks
        self.add_scan_callback = None
        
        self._create_widgets()
    
    def _create_widgets(self):
        """Create scan control widgets"""
        # Current data save directory
        self._create_directory_section()
        
        # Total scan time
        self._create_time_section()
        
        # File status message
        self._create_status_section()
        
        # Add to scan button
        self._create_add_scan_button()
    
    def _create_directory_section(self):
        """Create directory display section"""
        row = 19  # Based on original layout
        
        # Current data save directory
        scdir_txt = tk.Button(
            self.parent_frame, text="Current data save directory:", command=self.get_user_dir
        )
        scdir_txt.grid(row=row, column=10, sticky="w")
        
        pardir_out = tk.Label(self.parent_frame, textvariable=self.pardir)
        pardir_out.grid(row=row, column=11, sticky="w", columnspan=3)
    
    def _create_time_section(self):
        """Create time display section"""
        row = 21  # Based on original layout
        
        # Estimated total scan time
        tot_est_txt = tk.Label(self.parent_frame, text="Estimated total scan time:")
        tot_est_txt.grid(row=row, column=10, sticky="w")
        
        tot_est_val = tk.Label(self.parent_frame, textvariable=self.tot_time)
        tot_est_val.grid(row=row, column=11, sticky="w")
    
    def _create_status_section(self):
        """Create status message section"""
        # File status message
        from ..utils.validation_utils import limit_stringvar_length
        self.openfilemsg.trace("w", lambda *args: limit_stringvar_length(self.openfilemsg, 100))
        self.open_msg_label = tk.Label(self.parent_frame, textvariable=self.openfilemsg)
        self.open_msg_label.grid(row=59, column=0, columnspan=2)
    
    def _create_add_scan_button(self):
        """Create add to scan button"""
        # This will be called from the main setup frame
        pass
    
    def create_add_scan_button(self, row, col):
        """Create add to scan button at specified position"""
        add_scan_btn = tk.Button(
            self.parent_frame, text="Add to scan list", command=self._add_to_scan, width=50
        )
        add_scan_btn.grid(row=row + 5, column=col, columnspan=4)
    
    def get_user_dir(self):
        """Get user directory from PV"""
        self.pardir.set(self.pv_comm.getDir())
    
    def _add_to_scan(self):
        """Handle add to scan button click"""
        if self.add_scan_callback:
            self.add_scan_callback()
    
    def set_add_scan_callback(self, callback: Callable):
        """Set callback for add to scan button"""
        self.add_scan_callback = callback
    
    def set_file_status(self, message: str, color: str = "black"):
        """Set file status message"""
        self.openfilemsg.set(message)
        self.open_msg_label.config(fg=color)
    
    def set_total_time(self, time_str: str):
        """Set total scan time"""
        self.tot_time.set(time_str)
