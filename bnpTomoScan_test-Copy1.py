#!/APSshare/anaconda3/x86_64/bin/python

import sys, os
sys.path.append('/home/beams/USERBNP/scripts/')
from graceluo.imgProcessing import *
from graceluo.bnpTomoScan import bnpTomoScan
import numpy as np

'''
Define scan settings:

scanMode, str: ['coarse', 'fine', 'coarse_fine']
logfile, str: name of log file only, don't include path
userdir, str: parent directory of where the associated files are saved
angle_init, boolean: putting tomo_angle to 0 degree (True or False)
find_bbox, boolean: finding ROI within a coarse map (True or False)
smpInfo, str: information regarding the sample
coarse_w_h_sw_sh_dt, []: [width, height, stepsize_width, stepsize_height, dwelltime] for coarse scan
fine_w_h_sw_sh_dt, []: [width, height, stepsize_width, stepsize_height, dwelltime] for fine scan
orgPos_xyz, []: [x-center, y-center, z-position] at angle 0
coarse_scans, []: array of angles where coarse XRF maps are collected
fine_scans, []: [rotation_angle, x-center, y-center, z-position] for collecting fine XRF maps

When using COARSE scan mode:
    - REQUIRED parameters are: angle_init, coarse_w_h_sw_sh_dt, orgPos_xyz, coarse_scans
    - User need to decide whether to turn on/off angle_init. If angle_init is on, the scan will return to 0 degree after data collection
    - User need to define the parameter of coarse_w_h_sw_sh_dt
    - User need to define the parameter of orgPos_xyz
    - User need to define the parameter of coarse_scans (the array of angles)
    
When using FINE scan mode:
    - REQUIRED parameters are: fine_w_h_sw_sh_dt and fine_scans
    - Other parameters in the scan_setting dictionary are not used
    
When using COARSE-FINE scan mode:
    - Almost all parameters are REQUIRED except fine_scans.
    - OPTIONAL parameters: fine_scans
    - Parameters in fine_scans will be overwritten after performing ROI area estimation
    - find_bbox has to be turned on to allow ROI extraction
    - The workflow of coarse-fine scanning mode is:
        step 1, perform coarse scan at a given angle
        step 2, identify an ROI based on XRF intensity and return ROI area and center coordinates
        step 3, check if the ROI reasonable, if ROI area is less than 80% of the defined fine_w_h_sw_sh_dt, the function will proceed. Otherwise, terminate. A figure of ROI selection is save in the 'imgprocess' subfolder in userdir directory
        step 4, if ROI is reasonable, perform fine scan with updated x-center, y-center, z-pos from ROI analysis at this given anlge
        Iterate through step 1 to 4 for collection XRF projections at different angles

TO DO:
- correct_shifts, boolean: correcting shift between the scan that just finished and reference scan (refsc_shifts)
- test individual functions in bnpTomoScan.py
    Tested functions:
        openBeamBDA()
        closeBeamBDA()
        changeTomoRotate(-40)

'''
userdir = '/home/beams/USERBNP/scripts/graceluo/'
elm = 'K'
scan_mode_options = ['coarse', 'fine', 'coarse_fine']
scan_setting = {'scanMode':scan_mode_options[2],'logfile':'log.txt',
                'userdir':userdir,'angle_init':False,
                'find_bbox':True, 'BDAin':-9017,
#                'correct_shifts':False, 'refsc_shifts':None,
                'coarse_w_h_sw_sh_dt':[80, 12, 1, 1, 50],
                'fine_w_h_sw_sh_dt':[12, 10, 0.08, 0.08, 50],
                'orgPos_xyz':[46.8, -939, 580],
                'coarse_scans':[-40], # list of angles
                'fine_scans':[[12, 47.77, -939.5, 594.5]], # angle, x-, y-, z-pos
                'smpInfo':'sample_xxx', # a place for writing sample info in log file
                }
                
A = bnpTomoScan(scan_setting) # initialize bnpTomoScan object with scan_setting parameters
#A = bnpTomoScan.startScan()  # execute startScan will start the scanning execution

A.logfid.close()              # close log file before exit


