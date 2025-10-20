"""
Ptycho data handling module
Extracted from original setupFrame.py
"""

import os
import time
import tifffile
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from mic_vis.bnp.mda import get_mda_positioners


class PtychoDataHandler:
    """Handles Ptycho data loading and processing"""
    
    def __init__(self, pv_comm):
        self.pv_comm = pv_comm
        self.tiff = None
        self.tiff_filename = None
        self.mda_filepath = None
        self.x = None
        self.y = None
        self.file_z = None
        self.file_theta = None
        self.ptychirecon_dir_selected = True
        
    def choose_ptycho_folder(self, current_folder: str = "") -> str:
        """Choose Ptycho folder dialog"""
        if len(current_folder) < 5:
            if self.pv_comm.getDir() is not None:
                ptycho_dir = os.path.join(self.pv_comm.getDir(), "ptychi_recons")
                ptycho_dir = (
                    ptycho_dir if os.path.exists(ptycho_dir) else self.pv_comm.getDir()
                )
            else:
                ptycho_dir = "/mnt/micdata1/bnp/2021-3/"
        else:
            ptycho_dir = current_folder

        return ptycho_dir
    
    def get_scan_numbers(self, folder_path: str) -> List[str]:
        """Get list of scan numbers in folder"""
        if not os.path.exists(folder_path):
            return []
        
        return [i for i in os.listdir(folder_path) 
                if os.path.isdir(os.path.join(folder_path, i))]
    
    def list_recon_directory(self, folder_path: str, scan_number: str) -> Dict[str, Any]:
        """
        List reconstruction directory contents
        
        Args:
            folder_path: Path to ptycho folder
            scan_number: Selected scan number
            
        Returns:
            Dictionary containing available reconstruction options
        """
        selected_folder = os.path.join(folder_path, scan_number)
        
        if not os.path.exists(selected_folder):
            return {"error": f"Folder not found: {selected_folder}"}
        
        found_folders = [i for i in os.listdir(selected_folder) 
                        if os.path.isdir(os.path.join(selected_folder, i))]
        
        if "ML_recon" in selected_folder:
            # Ptychoshelf reconstruction
            self.ptychirecon_dir_selected = False
            show_type = "O_phase_roi"
            roi_folders = found_folders
            
            # Get reconstruction methods for first ROI
            recon_methods = []
            if roi_folders:
                roi_path = os.path.join(selected_folder, roi_folders[0])
                recon_methods = [i for i in os.listdir(roi_path) 
                               if os.path.isdir(os.path.join(roi_path, i))]
        else:
            # Ptychi reconstruction
            self.ptychirecon_dir_selected = True
            show_type = "object_ph"
            roi_folders = []
            recon_methods = found_folders
        
        return {
            "roi_folders": roi_folders,
            "recon_methods": recon_methods,
            "show_type": show_type,
            "ptychirecon_dir_selected": self.ptychirecon_dir_selected
        }
    
    def get_niter_list(self, folder_path: str, scan_number: str, 
                      roi_folder: str, recon_method: str) -> List[str]:
        """Get list of iteration files"""
        if self.ptychirecon_dir_selected:
            show_type = "object_ph"
        else:
            show_type = "O_phase_roi"
            
        result_folder = os.path.join(folder_path, scan_number, roi_folder, 
                                   recon_method, show_type)
        
        if not os.path.exists(result_folder):
            return []
        
        ext = ".tiff"
        niter_list = [i for i in os.listdir(result_folder) 
                     if i[-len(ext):] == ext]
        niter_list.sort(key=lambda x: int(x.split(".tiff")[0].replace(show_type+"_Niter", "")))
        
        return niter_list
    
    def _update_ptycho_mda_filepath(self, folder_path: str, scan_number: str):
        """Update MDA file path"""
        if self.ptychirecon_dir_selected:
            scan_num = int(scan_number.replace("S", ""))
            mda_dir = os.path.join(folder_path.split("/ptychi_recons")[0], "mda")
        else:
            scan_num = int(scan_number.replace("fly", ""))
            mda_dir = os.path.join(folder_path.split("/results/ML_recon")[0], "mda")
        
        # Find MDA file containing scan number
        if os.path.exists(mda_dir):
            mda_files = [i for i in os.listdir(mda_dir) if i.endswith(".mda")]
            for mda_file in mda_files:
                if str(scan_num) in mda_file:
                    self.mda_filepath = os.path.join(mda_dir, mda_file)
                    return
        
        self.mda_filepath = None
    
    def _update_ptycho_tiff_filepath(self, folder_path: str, scan_number: str,
                                   roi_folder: str, recon_method: str, niter_file: str):
        """Update TIFF file path"""
        if self.ptychirecon_dir_selected:
            show_type = "object_ph"
        else:
            show_type = "O_phase_roi"
            
        self.tiff_filename = os.path.join(folder_path, scan_number, roi_folder,
                                        recon_method, show_type, niter_file)
    
    def load_ptycho_scan(self, folder_path: str, scan_number: str,
                        roi_folder: str, recon_method: str, niter_file: str) -> Dict[str, Any]:
        """
        Load Ptycho scan data
        
        Args:
            folder_path: Path to ptycho folder
            scan_number: Selected scan number
            roi_folder: Selected ROI folder
            recon_method: Selected reconstruction method
            niter_file: Selected iteration file
            
        Returns:
            Dictionary containing scan data and metadata
        """
        self._update_ptycho_tiff_filepath(folder_path, scan_number, roi_folder, 
                                        recon_method, niter_file)
        self._update_ptycho_mda_filepath(folder_path, scan_number)
        
        # Check file modification time
        t_diff = 10
        fmtime = os.path.getmtime(self.tiff_filename)
        ctime = time.time()
        
        if (ctime - fmtime) < t_diff:
            return {"error": "File not ready, getting update from other process"}
        
        try:
            self.tiff = tifffile.imread(self.tiff_filename)
            
            # Handle uint data normalization
            if self.tiff.dtype.kind == "u":
                vmax = np.iinfo(self.tiff.dtype).max
                norm = self.tiff.astype(np.float32) / vmax
                self.tiff = norm
            
            # Get position data from MDA file
            if self.mda_filepath and os.path.exists(self.mda_filepath):
                positioners = get_mda_positioners(self.mda_filepath)
                self.y = np.linspace(positioners["y_pos"][1], positioners["y_pos"][-1], 
                                   self.tiff.shape[0])
                self.x = np.linspace(positioners["x_pos"][0], positioners["x_pos"][-1], 
                                   self.tiff.shape[1])
                self.file_z = positioners["z_pos"]
                self.file_theta = positioners["theta_pos"]
                
                return {
                    "tiff": self.tiff,
                    "x": self.x,
                    "y": self.y,
                    "file_z": self.file_z,
                    "file_theta": self.file_theta,
                    "success": True
                }
            else:
                return {"error": f"MDA file not found: {self.mda_filepath}"}
                
        except Exception as e:
            return {"error": f"Having trouble opening {self.tiff_filename}: {str(e)}"}
