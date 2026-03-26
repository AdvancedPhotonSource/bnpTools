"""
Created on Wed Oct 27 10:28:14 2021

@author: graceluo

Create and get PVobjects

"""
#!/home/beams/USERBNP/.conda/envs/py36/bin/python

import epics, sys, datetime
import numpy as np
from misc import getCurrentTime
import epics.devices
from epics import caput, caget

sscan = "9idbBNP:"
inner = "scan1"
outer = "scan2"
saveData_basePV = "9idbBNP"
# xrf_stp = 'bnpXP3:det1:Acquire' 
# plgn_stp = 'bnpXP3:HDF1:Capture'
# xrf_status= 'bnpXP3:det1:DetectorState_RBV' 
# plgn_save = 'bnpXP3:HDF1:WriteFile'
# plgn_status = 'bnpXP3:HDF1:WriteFile_RBV'
# xrf_mode = 'bnpXP3:det1:TriggerMode'

xrf_stp = '9idbXMAP:netCDF1:StopAll' 
plgn_stp = '9idbXMAP:netCDF1:Capture'
xrf_status= '9idbXMAP:netCDF1:DetectorState_RBV' 
plgn_save = '9idbXMAP:netCDF1:WriteFile'
plgn_status = '9idbXMAP:netCDF1:WriteFile_RBV'
xrf_mode = '9idbXMAP:CollectMode'

# Eiger object is create for accessing camera related attributes
class eiger(object):
    def __init__(self, cam_pv_str, file_pv_str):
        self.pvstr = cam_pv_str
        self.cam = epics.devices.AD_Camera(cam_pv_str)
        self.fileIO = epics.devices.AD_FilePlugin(file_pv_str)
        
    def setNumTriggers(self, numTriggers):
        caput('%sNumTriggers'%self.pvstr, numTriggers)
    
    def getNumTriggers(self):
        return caget('%sNumTriggers'%self.pvstr)

class pvObject(object):
    def __init__(self, pv_str, pv_key, onchange_callback=False):
        self.pv = epics.PV(pv_str)
        self.pvname = pv_key
        self.putvalue = self.pv.value
        self.put_complete = 0
        self.motor_ready = 1
        self.time_pre = None    #datetime of PV when connected or previous value change
        self.time_delta = 0    #time difference in sec btw its value change
        if onchange_callback:
            self.pv.add_callback(self.onChanges)
        
            
    def onPutComplete(self, pvname=None,  **kws):
        sys.stdout.write('%s: Finish updating PV %s with value of %s\n'\
                          %(getCurrentTime(), self.pvname, str(self.putvalue)))
        self.put_complete = 1
        
    def onChanges(self, pvname=None, **kws):
            
        if self.time_pre is None: 
            self.time_pre = datetime.datetime.now()
        else:
            curtime = datetime.datetime.now()
            self.time_delta = (curtime-self.time_pre).seconds
            self.time_pre = curtime
            
        sys.stdout.write('%s: previous time:%s, delta time:%s\n'
                         %(getCurrentTime(), self.time_pre, self.time_delta))

        
    def put_callback(self, v = None):
        self.put_complete = 0
        if v is not None:
            self.putvalue = v
            self.pv.put(self.putvalue, callback=self.onPutComplete)

    def motorReady(self, rqspv, tolerance = 4e-2):
        rqsvalue = np.round(rqspv.value, 2)
        if abs((np.round(self.pv.value, 2) - rqsvalue)) < tolerance:
            self.motor_ready = 1
        else:
            rqspv.put(rqsvalue)
            self.motor_ready = 0
        
        
def definePVs():
    pvs = {'x_center_Rqs':'9idbTAU:SM:PX:RqsPos', 'x_center_Act':'9idbTAU:SM:PX:ActPos',
            'y_center_Rqs':'9idbTAU:SY:PY:RqsPos', 'y_center_Act':'9idbTAU:SY:PY:ActPos',
            'z_value_Rqs':'9idbTAU:SM:SZ:RqsPos', 'z_value_Act':'9idbTAU:SM:SZ:ActPos',
            'tomo_rot_Rqs':'9idbTAU:SM:CT:RqsPos', 'tomo_rot_Act':'9idbTAU:SM:CT:ActPos',
            'sm_rot_Rqs':'9idbTAU:SM:ST:RqsPos', 'sm_rot_Act':'9idbTAU:SM:ST:ActPos',
            'x_width':f'{sscan}{inner}.P1WD', 'y_width':f'{sscan}{outer}.P1WD',
            'x_step':f'{sscan}{inner}.P1SI', 'y_step':f'{sscan}{outer}.P1SI',
            'dwell':'9idbBNP:scanTran3.C', 'BDA_pos':'9idbTAU:UA:UX:RqsPos',
            'det_time':'9idbBNP:3820:ElapsedReal', '1D_time':'9idbBNP:scanTran4.F',
            'mcs_stp':'9idbBNP:3820:StopAll', 'mcs_status':'9idbBNP:3820:Acquiring',

            'xmap_stp':f'{xrf_stp}', 'netCDF_stp':f'{plgn_stp}',
            'xmap_status':f'{xrf_status}', 'netCDF_save':f'{plgn_save}',
            'netCDF_status':f'{plgn_status}',
            'collect_mode':f'{xrf_mode}',

            'y_motor_ready':'9idbTAU:SY:Ps:Ready', 'xztp_motor_ready':'9idbTAU:SM:Ps:Ready',
            'x_piezo_val':'9idbTAU:M7009.VAL', 'y_piezo_val':'9idbTAU:M7010.VAL',
            'scan2Record':f'{sscan}{outer}',
            
            'x_motorMode':'9idbTAU:SM:Ps:xMotionChoice.VAL',
            'y_motorMode':'9idbTAU:SY:Ps:yMotionChoice.VAL',
            'x_updatecenter':f'{sscan}{inner}.P1CP', 'y_updatecenter':f'{sscan}{outer}.P1CP',
            # 'x_setcenter':'9idbBNP:aoRecord11.PROC', 'y_setcenter':'9idbBNP:aoRecord12.PROC',
            'piezo_xCenter':'9idbTAU:SM:Ps:xCenter.PROC',
            'piezo_yCenter':'9idbTAU:SY:Ps:yCenter.PROC',
            'tot_lines':f'{sscan}{outer}.NPTS', 'cur_lines':f'{sscan}{outer}.CPT',
            'tot_pts_perline':f'{sscan}{inner}.NPTS',
            
            'CryoCon1:In_1':'9idbCRYO:CryoCon1:In_1:Temp.VAL',
            'CryoCon1:In_3':'9idbCRYO:CryoCon1:In_3:Temp.VAL',
            'CryoCon1:In_2':'9idbCRYO:CryoCon1:In_2:Temp.VAL',
            'CryoCon3:In_2':'9idbCRYO:CryoCon3:In_2:Temp.VAL',
            'CryoCon3:Loop_2':'9idbCRYO:CryoCon3:Loop_2:SetControl.VAL',

            'run':f'{sscan}{outer}.EXSC', 'wait':f'{sscan}{outer}.WAIT', 'wait_val':f'{sscan}{outer}.WCNT',
            'pause':f'{sscan}{inner}.PAUS', 'abort':f'{sscan}AbortScans.PROC', 
            'msg1d':f'{sscan}{inner}.SMSG',
            'fname_saveData':f'{saveData_basePV}:saveData_fileName',
            'filesys':f'{saveData_basePV}:saveData_fileSystem',
            'subdir':f'{saveData_basePV}:saveData_subDir',
            'nextsc':f'{saveData_basePV}:saveData_scanNumber',
            'basename':f'{saveData_basePV}:saveData_baseName',
            
            }
    return pvs
 
def scan2RecordDetectorTrigerPVs():
    pvs = {'scan1':f'{sscan}{inner}.EXSC',
           'eigerAcquire':'2iddEGR:cam1:Acquire',
           'eigerFileCapture':'2iddEGR:HDF1:Capture'}
    return pvs
    
def getEiger():
    # create Eiger cam record
    e = eiger('2iddEGR:cam1:', '2iddEGR:HDF1:')
    return e
    
#    pvs = {'test1':'2idbleps:userTran2.CMTA', 'test2':'2idbleps:userTran2.CMTB', 'test3':'2idbleps:userTran2.CMTC',
#           'test4':'2idbleps:userTran2.CMTD', 'test5':'2idbleps:userTran2.CMTE', 'test6':'2idbleps:userTran2.CMTF',
#           'test7':'2idbleps:userTran2.CMTG'}


def getPVobj():
    pvObjs = {}
    pvs = definePVs()
    for k, v in pvs.items():
        if 'Record' not in k:
            pv_obj = pvObject(v, k, onchange_callback=True if f'{sscan}{outer}.CPT'==v else False)
            pvObjs.update({k: pv_obj})
        else:
            pvObjs.update({k:epics.devices.Scan(v)})
    return pvObjs
