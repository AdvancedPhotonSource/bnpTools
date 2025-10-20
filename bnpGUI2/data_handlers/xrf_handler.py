"""
XRF data handling module
Extracted from original setupFrame.py
"""

import os
import time
import h5py
import numpy as np
from typing import Dict, List, Optional, Tuple, Any


class XRFDataHandler:
    """Handles XRF data loading and processing"""
    
    def __init__(self, pv_comm):
        self.pv_comm = pv_comm
        self.h5_file = None
        self.h5_filename = None
        self.x = None
        self.y = None
        self.file_z = None
        self.file_theta = None
        
    def choose_xrf_folder(self) -> str:
        """Choose XRF folder dialog"""
        if self.pv_comm.getDir() is not None:
            initial_dir = os.path.join(self.pv_comm.getDir(), "img.dat")
            initial_dir = (
                initial_dir if os.path.exists(initial_dir) else self.pv_comm.getDir()
            )
        else:
            initial_dir = "/mnt/micdata1/2idd"

        return initial_dir
    
    def get_h5_files(self, folder_path: str) -> List[str]:
        """Get list of H5 files in folder"""
        if not os.path.exists(folder_path):
            return []
        
        ext = ".h5"
        return [i for i in os.listdir(folder_path) if i[-len(ext):] == ext]
    
    def load_xrf_scan(self, folder_path: str, filename: str) -> Dict[str, Any]:
        """
        Load XRF scan data
        
        Args:
            folder_path: Path to folder containing H5 files
            filename: Name of H5 file to load
            
        Returns:
            Dictionary containing scan data and metadata
        """
        self.h5_filename = os.path.join(folder_path, filename)
        
        # Close previous file
        if self.h5_file:
            try:
                self.h5_file.close()
            except:
                pass
        
        # Check file modification time
        t_diff = 10
        fmtime = os.path.getmtime(self.h5_filename)
        ctime = time.time()
        
        if (ctime - fmtime) < t_diff:
            return {"error": "File not ready, getting update from other process"}
        
        try:
            self.h5_file = h5py.File(self.h5_filename, "r")
            
            # Get detector names
            dets = (
                self.h5_file["/MAPS/channel_names"][:].astype(str).tolist()
                + self.h5_file["/MAPS/scaler_names"][:].astype(str).tolist()
            )
            
            # Get axis data
            self.x = self.h5_file["/MAPS/x_axis"][()]
            self.y = self.h5_file["/MAPS/y_axis"][()]
            
            # Get PV data
            pvlist = self.h5_file["/MAPS/extra_pvs"][0].astype(str).tolist()
            pvval = self.h5_file["/MAPS/extra_pvs"][1].astype(str).tolist()
            
            if len(pvlist) >= 5:
                try:
                    self.file_z = float(
                        pvval[pvlist.index(self.pv_comm.pvs["z_value_Act"].pv.pvname)]
                    )
                    self.file_theta = float(
                        pvval[pvlist.index(self.pv_comm.pvs["sm_rot_Act"].pv.pvname)]
                    )
                except (ValueError, KeyError):
                    self.file_z = None
                    self.file_theta = None
            else:
                self.file_z = None
                self.file_theta = None
            
            return {
                "detectors": dets,
                "x": self.x,
                "y": self.y,
                "file_z": self.file_z,
                "file_theta": self.file_theta,
                "h5_file": self.h5_file,
                "success": True
            }
            
        except Exception as e:
            return {"error": f"Having trouble opening {self.h5_filename}: {str(e)}"}
    
    def get_plot_data(self, detector_index: int) -> Optional[np.ndarray]:
        """Get plot data for specific detector"""
        if not self.h5_file:
            return None
            
        try:
            elm_scalers = np.vstack(
                (self.h5_file["/MAPS/XRF_roi"][:], self.h5_file["/MAPS/scalers"][:])
            )
            return elm_scalers[detector_index]
        except:
            return None
    
    def close(self):
        """Close H5 file"""
        if self.h5_file:
            try:
                self.h5_file.close()
            except:
                pass
