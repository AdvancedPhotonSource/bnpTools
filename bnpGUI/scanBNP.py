'''
Control PV based on type of measurements

'''
#!/home/beams/USERBNP/.conda/envs/py36/bin/python

import os, time, sys
import numpy as np
from mic_vis.bnp.mda import get_mda_positioners
import tifffile

from imgProcessing import getROIcoordinate_data, getElmMap

def parmLabelToPVdict():
    d = {'width':'x_width', 'height':'y_width', 'w_step':'x_step',
         'h_step':'y_step', 'dwell':'dwell', 'x_scan':'x_center_Rqs', 
         'y_scan':'y_center_Rqs', 'z_scan': 'z_value_Rqs', 'target_theta':'sm_rot_Rqs'}
    return d

def xrfSetup(pvComm, scandic):
    d = parmLabelToPVdict()
    parms = ['width', 'height', 'w_step', 'h_step', 
             'dwell', 'y_scan', 'x_scan']
    parm_label = [d[s] for s in parms]
    parm_value = [float(scandic[s]) for s in parms]
    pvComm.writeScanInit('XRF', scandic['smpName'], str(scandic))
    pvComm.blockBeamBDA(scandic['bda'])
    # pvComm.changeXYcombinedMode()
    pvComm.changeXtoCombinedMode()
    pvComm.assignPosValToPVs(parm_label, parm_value)
    return getMotorList(scandic)

def getMotorList(scandic):
    p = ['target_theta', 'z_scan', 'y_scan', 'x_scan']
    motorlabel = ['sm_rot', 'z_value', 'y_center', 'x_center']
    mtolerance = [0.1, 0.5, 0.1, 0.1]
    mlist = []
    for p_, ml_, mt_ in zip(p, motorlabel, mtolerance):
        mlist.append((ml_, float(scandic[p_]), mt_))
    return mlist

def scanStart(pvComm, bda):
    pvComm.changeXtoPiezolMode()
    pvComm.openBeamBDA(bda)

def scanFinish(pvComm, bda):
    time.sleep(5)
    pvComm.blockBeamBDA(bda)
    pvComm.changeXtoCombinedMode()
    pvComm.centerPiezoY(waittime=3)   #at 2idd, stage issue, work around
    pvComm.centerPiezoY(waittime=3)   #at 2idd, stage issue, work around
    time.sleep(0.5)

def generate_ptycho_fpath(coarse_sc, fdir, recon_method = 'Ndp64_LSQML_s800_gaussian_p10_cp_mm_opr3_ic_pc0_f_ul2'):
    coarse_num = coarse_sc.split('.mda')[0].replace('bnp_fly', '')
    fpath = os.path.join(fdir, f'ptychi_recons/S{coarse_num}/{recon_method}/object_ph/object_ph_Niter200.tiff')
    return fpath

def generate_mda_fpath(coarse_sc, fdir):
    fpath = os.path.join(fdir, f'mda/{coarse_sc}')
    return fpath
    
def fileReady(coarse_sc, fdir, tlim = 30, ptycho = False, recon_method = 'Ndp64_LSQML_s800_gaussian_p10_cp_mm_opr3_ic_pc0_f_ul2'):
    if ptycho:
        fpath = generate_ptycho_fpath(coarse_sc, fdir, recon_method)
    else:
        fpath = os.path.join(fdir, 'img.dat/%s.h5'%(coarse_sc))

    if os.path.exists(fpath):
        fmtime = os.path.getmtime(fpath)
        tdiff = time.time() - fmtime
        if tdiff > tlim:
            return 1
        else:
            sys.stdout.write('Waiting for coarse scan file %s.h5 to be ready,'\
                     ' file modified time: %d, time difference: %d \n'\
                     %(coarse_sc, fmtime, tdiff))
            return 0
    else:
        sys.stdout.write('File %s not exisit\n'%fpath)
        return 0
    
def imgProgFolderCheck(fdir):
    img_path= os.path.join(fdir,'imgProg')
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    return img_path

def getCoordinate(pvComm, coarse_sc, scandic, n_std = 2):
    fready = fileReady(coarse_sc, pvComm.userdir)
    coarse_h5path = os.path.join(pvComm.userdir, 'img.dat/%s.h5'%(coarse_sc))
    if fready:
        imgfolder = imgProgFolderCheck(pvComm.userdir)   
        imgpath = os.path.join(imgfolder, 'bbox_%s.png'%(coarse_sc))
        print(imgpath)
        elmmap = getElmMap(coarse_h5path, scandic['elm'])
        
        mask = np.ones(elmmap[0].shape)
        if scandic['use_mask']:
            maskmap = getElmMap(coarse_h5path, scandic['mask_elm'])
            mask = maskmap < (np.mean(maskmap) + n_std * np.std(maskmap.ravel()))
        m = elmmap[0] * mask
        
        x, y, w, h = getROIcoordinate_data(m, elmmap[1], elmmap[2], 
                                           n_cluster = scandic['n_clusters'],
                                           sel_cluster = scandic['sel_cluster'],
                                           figpath = imgpath)
        return np.round(x,2), np.round(y,2)
    else:
        return None
    

def get_ptycho_data_positions(coarse_sc, fdir):
    
    mda_fpath = generate_mda_fpath(coarse_sc, fdir)
    ptycho_fpath = generate_ptycho_fpath(coarse_sc, fdir)

    print(mda_fpath)
    print(ptycho_fpath)

    if os.path.exists(mda_fpath) and os.path.exists(ptycho_fpath):
        ptycho_data = tifffile.imread(ptycho_fpath)
        if ptycho_data.dtype.kind == "u":
            vmax = np.iinfo(ptycho_data.dtype).max
            norm = ptycho_data.astype(np.float32) / vmax
            ptycho_data = norm
            ptycho_data = ptycho_data * (-1)

        if os.path.exists(mda_fpath):
            positioners = get_mda_positioners(mda_fpath)
            y = np.linspace(positioners["y_pos"][1], positioners["y_pos"][-1], ptycho_data.shape[0])
            x = np.linspace(positioners["x_pos"][0], positioners["x_pos"][-1], ptycho_data.shape[1])
        
        return ptycho_data, x, y
    else:
        return None


def getCoordinate_ptycho(pvComm, coarse_sc, scandic, n_std = 2):
    fready = fileReady(coarse_sc, pvComm.userdir, ptycho = True)
    
    if fready:
        imgfolder = imgProgFolderCheck(pvComm.userdir)   
        imgpath = os.path.join(imgfolder, 'bbox_%s.png'%(coarse_sc))
        print(imgpath)
        ptycho_data, x_pos, y_pos = get_ptycho_data_positions(coarse_sc, pvComm.userdir)
        # ptycho_data_norm = (ptycho_data - np.min(ptycho_data)) / (np.max(ptycho_data) - np.min(ptycho_data))
        
        x, y, w, h = getROIcoordinate_data(ptycho_data, x_pos, y_pos, 
                                           n_cluster = scandic['n_clusters'],
                                           sel_cluster = scandic['sel_cluster'],
                                           figpath = imgpath)
        return np.round(x,2), np.round(y,2)
    else:
        return None
    
    
            




