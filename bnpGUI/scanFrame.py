"""
Created on Tue Aug  3 11:50:01 2021

@author: graceluo

Construct scan frame
"""

#!/home/beams/USERBNP/.conda/envs/py36/bin/python

import tkinter as tk
from tkinter import ttk
from scanList import scanList
from misc import checkEntryDigit
#from bnpScan import bnpScan, 
from pvComm import pvComm, pvCommsubclass
from scanBNP import xrfSetup, scanStart, scanFinish, getCoordinate_2, getMotorList, xanes_ps_c, xanes_ps_n, xrfSetup_fromxanes#, getCoordinate_2
from logger import stdoutToTextbox
import time, datetime
import pandas as pd
import epics as PV
import os

class scanFrame():
    
    def checkUserDir(self):
        if self.pvComm.getDir() != self.pvComm.userdir:
            self.pvComm = pvComm()
            
    def updateRecord(self):
        self.slist.sclist.item(self.record, text='', 
                               values = tuple(self.recordval))
    
    def scanClick(self):
        if self.scan_btn['state'] == tk.ACTIVE:
            self.scan_btn['state'] = tk.DISABLED
            self.scanclick = True
    
    def scanSetup(self):
        #xyl: add a flog to determine whether the next one is the same or not 
        self.flag, self.record, self.recordval = self.slist.searchQueue()  #values for the first in queue and change it to 'scanning'
    #recordval is the parameters for the first scan  # a function from scanList
        if self.record is not None:
            self.pause_btn['state'] = tk.NORMAL
            self.slist.pbarInit()
            self.scan_start_time = 0
            self.scdic = {u:self.recordval[i] for i, u in enumerate(self.slist.sclist_col)}  #scans in sclist
            self.scdic.update({'bda':float(self.bda.get())})
            self.checkUserDir()
            self.stype = self.scdic['scanType']
            self.recordval[0] = self.pvComm.nextScanName()
            self.coordsReady = False
            self.eigerReady = False
            
            
            if self.stype =='Coarse':
                self.coarse_scnum = self.recordval[0]
            
            elif self.stype == 'Fine':
                time.sleep(5)  #give 1 min sleep before start 'Fine' scan
                print('wait 5s before Fine scan for getting .h5 file')
                self.cur_ang = round(PV.caget('9idbTAU:SM:ST:ActPos'))  #just get int
                self.angle_list.append(self.cur_ang)
                print('current angle: {self.cur_ang} before check fine scan coordinates')
                self.checkFineScanCoord()
            #----------------xyl: add scan setup for xanes-----------
            elif self.stype == 'XANES (fixed region)' and self.flag == 2:  # means this one is xanes, previous one is xrf, needs to change to step mode
                self.coordsReady = True
                self.motors = xanes_ps_n(self.pvComm,self.scandic)   # change mono to auto, drive 1 to mono, read 1 to NA, center, width, step, dwell
                self.monitormsg.set('wait for motors to be ready')
                
            elif self.stype == 'XANES (fixed region)' and self.flag == 0: # this one and previous one are xanes, continue step mode
                self.coordsReady = True
                self.motors = xanes_ps_c(self.pvComm,self.scandic)   # only change center, width, step, dwell
                self.monitormsg.set('wait for motors to be ready')
                
            elif self.stype == 'XRF' and self.flag == 1:  #this one is xrf, previous one is xanes, needs to change to fly mode
                self.coordsReady = True
                self.motors = xrfSetup_fromxanes(self.pvComm,self.scandic)
                self.monitormsg.set('wait for motors to be ready')
                
            else:  # for continuous xrf
                self.coordsReady = True
                self.motors = xrfSetup(self.pvComm, self.scdic) # return a list of tuples
                self.monitormsg.set('wait for motors to be ready')
            
            self.pending = True
        else:
            self.scanclick = False
            self.scan_btn['state'] = tk.NORMAL
            self.pause_btn['state'] = tk.DISABLED
            self.monitormsg.set('Not scanning, not active')
        
    
    def updateXYZcoor(self, fine_corr):
        for i, s_ in enumerate(['x_scan', 'y_scan']):
            self.scdic[s_] = fine_corr[i]
            self.recordval[self.slist.sclist_col.index(s_)] = '%.2f'%(fine_corr[i])
        self.updateRecord()
    '''
    def checkFineScanCoord(self):
        fine_coor = getCoordinate(self.pvComm, self.coarse_scnum, 
                                  self.scdic)
        if fine_coor is not None:
            self.updateXYZcoor(fine_coor)
            self.motors = xrfSetup(self.pvComm, self.scdic)
            self.coordsReady = True
            self.pending = True
        else:
            self.monitormsg.set('Waiting for coordinates for fine scans')
    '''
    
#--------------------------xyl-------------------------

    
    def checkFineScanCoord(self):
        coarse_scan_num = pvCommsubclass().scan_mda()  #bnp_fly_xxxx
        coarse_scan_num_h5 = coarse_scan_num + '.mda.h5'
        fine_coor = getCoordinate_2(coarse_sc=coarse_scan_num_h5, 
                                      scandic=self.scdic, flag=self.scdic['flag'])  #function from scanBNP
        if fine_coor is not None:
            self.x_scan_list.append(fine_coor[0])
            self.y_scan_list.append(fine_coor[1])    #make a dataframe to record the fine scan x,y positions
            self.real_ang_test.append(fine_coor[2])
            self.updateXYZcoor(fine_coor)  #update the value of 'x_scan' and 'y_scan' from coarse scan
            self.motors = xrfSetup(self.pvComm, self.scdic)   #from scanBNP: return a turple list: 
                                                              #[('sm_rot',value of 'target_theta',0.1),
                                                              # ('z_value', value of 'z_scan',0.5),
                                                              # ('y_center', value of 'y_scan', 0.1),
                                                              # ('x_center', value of 'x_scan', 0.1)]
            self.coordsReady = True
            self.pending = True
        else:# for the case that no x y value inputs into fine scan
            #cur_ang = PV('9idbTAU:SM:ST:ActPos').get()  
            #print(f'No x and y coordinates, curent angle:{self.cur_ang}')
            self.monitormsg.set(f'No x and y coordinates, curent angle:{self.cur_ang}')
            scanFinish(self.scdic, self.pvComm, float(self.bda.get()))
            self.recordval[1] = 'No x,y value'
            self.x_scan_list.append('No x found')
            self.y_scan_list.append('No y found')
            self.updateRecord()
            self.motorReady = 0
            self.pbarscval.set(1)
            self.slist.pbarInit()
            self.scan_start_time = 0
            self.scandone_var.set(False)
            self.scanSetup()
        print(self.x_scan_list)
        self.df = pd.DataFrame(list(zip(self.angle_list,self.x_scan_list,self.y_scan_list)),columns=['Angle','X_pos','Y_pos'])
        #self.df.to_csv('/mnt/micdata1/bnp/2023-1/test_gui_xyl/fine_x_y_coordinates.csv')
        self.df.to_csv(f'{self.user_folder}/fine_x_y_coordinates.csv')           
#------------------------xyl-------------------------------------------------------------

    def checkMotorReady(self):
        # Add timeout parameters
        if len(self.motors) > 0:
            mn, mv, mt = self.motors[0]
            act = '%s_Act'%mn
            rqs = '%s_Rqs'%mn
            self.pvComm.assignSinglePV(rqs, mv)
            mready = self.pvComm.motorReady(mn, mt)
            if mready == 1:
                self.motors.pop(0)
        else:
            self.pvComm.setXYcenter()
            self.pvComm.centerPiezoXY()
            time.sleep(0.5)
            self.pvComm.centerPiezoXY()
            time.sleep(0.5)
            xztp_ready = self.pvComm.motorReady_XZTP()
            motor_diff = self.pvComm.sumMotorDiff(getMotorList(self.scdic))
            print('Motor difference: %.2f'%(motor_diff))
            MOTOR_DIFF_MIN = 0.5
            if (xztp_ready) & (motor_diff < MOTOR_DIFF_MIN):
                self.motorReady = 1
            else:
                self.motors = getMotorList(self.scdic)
                
    def checkEigerReady(self):
        # check detector triger on scan record
        self.pvComm.updateDetectorTriger(int(self.scdic['ptycho']))
        
        # if ptycho is checked
        if self.scdic['ptycho']:
            print(self.scdic)
            acquireTime = self.scdic['dwell']/1e3  # s
            trig_mode = 2   #external_series
            num_trig = 1    # when trig_mode = 1, it needs 1 trigger per line
            num_img = self.pvComm.getNumPtsPerLine()
            filename = self.recordval[0].replace('.mda', '')
            print(acquireTime, num_img)
            
            self.pvComm.updateEigerCam(acquireTime, num_img, trig_mode, num_trig)
            self.pvComm.updateEigerFileIO(filename, num_img)
            self.eigerReady = True
        else:
            self.eigerReady = True
    
    def scanDone(self, *args, **kwargs):
        if self.scandone_var.get():
            scanFinish(self.scdic, self.pvComm, float(self.bda.get()))
            self.recordval[1] = 'done'
            self.updateRecord()
            self.motorReady = 0
            self.pbarscval.set(1)
            self.slist.pbarInit()
            self.scan_start_time = 0
            self.scandone_var.set(False)
    
    def scanExec(self):
        # print('scan exect')
        scanStart(self.scdic, self.pvComm, float(self.bda.get()))  #in scanBNP.py, change x to piezo mode and move bda to open
        # if status:
        self.pbarscval.set(0.0)
        self.monitormsg.set('Motor ready')
        time.sleep(0.5)
        self.pending = False
        self.monitormsg.set('scanning')
        self.recordval[1] = 'scanning'
        self.updateRecord()
        self.pvComm.pvs['run'].pv.put(1)
        self.pvComm.initCurLineTimer()
        self.scanning = True
        # else:
        #     # self.monitormsg.set('Fail to center XY piezo motors... tyring again')
        #     print('Fail to center XY piezo motors... batch scan paused... check piezo motors')
        #     self.pauseClick()

    def scanMonitor(self, *args, **kwargs):
        ms = 1000
        
        if self.pause & self.scanning:
            if self.pvComm.pvs['wait_val'].pv.get() == 0:
                self.pvComm.scanPause()
                self.monitormsg.set('Scan Pause with 1 wait flag, waiting for current line to finish')
                time.sleep(1)
                
            elif self.pvComm.pvs['msg1d'].pv.get() == 'SCAN Complete':
                if self.ycenter_check:
                    self.pvComm.centerPiezoY()
                    self.checkYCenterValue()
                elif self.detector_resetting:   # it will enter here when detector reset is successful
                    self.checkDetectorStatus()
                else:
                    self.abort_btn['state'] = tk.NORMAL
                    self.abortall_btn['state'] = tk.NORMAL
                    self.resume_btn['state'] = tk.NORMAL  
                    
            elif self.pvComm.pvs['msg1d'].pv.get() != 'SCAN Complete':
                if self.ycenter_check:
                    self.checkDetectorStatus()
                if self.detector_resetting:
                    self.det_reset_attemp += 1
                    self.pvComm.resetDetector(self.scdic['ptycho'])
                    self.checkDetectorStatus()

        
        # when pause and not scanning 
        elif self.pause & self.motorReady: pass
        elif self.pause & self.coordsReady: pass
        elif self.pause & self.eigerReady: pass
    
        elif not self.pause:
            self.abortall_btn['state'] = tk.DISABLED
            if self.scanclick & self.pending:  #seems connect to click scan button
                if not self.coordsReady:
                    self.checkFineScanCoord()
                    ms = 2000
                elif not self.motorReady:
                    self.checkMotorReady()
                elif not self.eigerReady:
                    self.checkEigerReady()
                else:
                    self.scanExec()
                        
            elif self.scanclick & (not self.pending) & (not self.scanning):
                self.monitormsg.set('look for next scan')
                self.scanSetup()
                
            elif self.scanning:
                # if (not self.pvComm.pvs['pause'].pv.value | self.pvComm.pvs['wait'].pv.get() > 0):
                if not self.pvComm.pvs['pause'].pv.value:
                    self.monitormsg.set('scanning')
                    if self.scan_start_time == 0: 
                        self.scan_start_time = datetime.datetime.now()
                        self.det_reset_attemp = 0
                        self.cline = 0
                    self.pbarUpdate()
                    self.checkDetectorStatus()
                    self.logTempPV()
                    self.monitormsg.set('Scanning... will be done at: %s'%self.eta_str)
                    # self.checkYCenterValue() #probably not needed after piezo replacement
                    
                else:
                    self.monitormsg.set('scan is paused or has not started yet... check end station shutter')
                
                if self.pvComm.pvs['run'].pv.value == 0:
                    self.scanning = False
                    self.scandone_var.set(True)
                
        self.mmsg_label.after(ms, self.scanMonitor)
        
    def checkYCenterValue(self,max_ycenter = 31000):
        ycenter = abs(self.pvComm.getYCenterValue())
        if ycenter > max_ycenter:
            print('Current ycenter value %d is larger than expected (%d)'%(ycenter, max_ycenter))
            if not self.ycenter_check:
                self.ycenter_check = True
                self.pause = True   
        elif self.ycenter_check:
            self.ycenter_check = False
            self.pause = False
            self.pvComm.scanResume()
            
    
    def logTempPV(self):
        if self.logTemp.get():
            self.pvComm.logCryoTemp()

    def pbarUpdate(self):
        cline = self.pvComm.pvs['cur_lines'].pv.value
        tline = self.pvComm.pvs['tot_lines'].pv.value
        # time_delta = datetime.datetime.now() - self.scan_start_time
        # timePerLine = (time_delta.seconds) / float((cline+1))
        timePerLine = self.pvComm.pvs['cur_lines'].time_delta
        remain_st = timePerLine*(tline-cline)
        eta_time = pd.Timestamp.now() + pd.DateOffset(minutes = self.slist.getTotalTime(remaining_st=remain_st))
        self.eta_str = eta_time.strftime('%Y-%m-%d %X')
        self.pbarscmsg.set('%s [%d / %d] %.1f %%remaining' % (self.recordval[0],
                           cline, tline, 100-cline/tline*100))
        self.pbarscval.set(cline/tline*100)
        
    def checkDetectorStatus(self, extra_wait = 20, max_attemp = 5):
        if self.detectorMonitor.get():
            pause_status = self.pvComm.pvs["pause"].pv.value
            cline = self.pvComm.pvs["cur_lines"].pv.value

            # replace in the future, using 1D time instead
            time_check = round(self.pvComm.get1DTime()) + extra_wait
            self.detCheck_val.set('%d'%time_check)
            
            # time_check = float(self.detCheck_val.get())  # getting detCheck from user
            time_now = datetime.datetime.now()
            time_pre = self.pvComm.getCurLineTimeStamp()
            time_delta = (time_now-time_pre).seconds
            
            
            if not pause_status:
                if all([self.detector_resetting, 
                          self.pvComm.pvs['msg1d'].pv.get() == 'SCAN Complete']):
                    # print('in checkdetector, scan complete returns')
                    self.detector_resetting = False
                    self.pause = False
                    self.pvComm.initCurLineTimer()
                    self.pvComm.scanResume()
                    
                elif all([time_delta > time_check, 
                        self.det_reset_attemp < max_attemp]):
                    if not self.detector_resetting:
                        self.pause = True
                        self.detector_resetting = True
                        self.pvComm.scanPause()
                        time.sleep(0.5)
                        
                    # self.pvComm.resetDetector()  # put it here to handle the case when 1st line hangs
                    self.monitormsg.set('Scan hungs... Resetting detector')
                    print('line %d: time per line %.2f, reset when time larger than %.2f, number of attamp: %d'
                          %(self.cline, time_delta, time_check, self.det_reset_attemp))
                        
                elif time_delta < time_check:
                    self.cline = cline
                    self.det_reset_attemp = 0
                elif self.det_reset_attemp > max_attemp:
                    print('Scan hungs... Reach detector reset limit... Need to try reset manually')

            else:
                self.detector_resetting = False
                self.pause = False
                self.pvComm.initCurLineTimer()
                self.pvComm.scanResume()
                    

    def pauseClick(self):
        print('Pause scan thread pressed')
        self.pause = True
        self.pause_btn['state'] = tk.DISABLED
        if not self.scanning:
            self.resume_btn['state'] = tk.NORMAL
            self.abort_btn['state'] = tk.NORMAL
        
    def resumeClick(self):
        print('Resume scan thread pressed')
        self.pause_btn['state'] = tk.NORMAL
        self.resume_btn['state'] = tk.DISABLED
        self.abort_btn['state'] = tk.DISABLED
        self.abortall_btn['state'] = tk.DISABLED
        self.pause = False

        if self.scanning:
            self.pvComm.scanResume()
    
    def abortSingleClick(self):
        print('Abort single clicked, aborting current scan')
        self.abort_btn['state'] = tk.DISABLED
        self.resume_btn['state'] = tk.DISABLED
        self.abortall_btn['state'] = tk.DISABLED
        self.pause = False
        self.pause_btn['state'] = tk.NORMAL
        # self.abortsingle = True
        if self.scanning:
            self.pvComm.scanAbort()
            self.pvComm.scanResume()
            self.scanning = False
            self.scandone_var.set(True)
            self.recordval[1] = 'abort'
            self.updateRecord()
        elif (not self.scanning) & (not self.abortall):
            self.recordval[1] = 'abort'
            self.updateRecord()
            self.pending = False
            self.scanSetup()

    def abortAllClick(self):
        print('Abort all clicked')
        self.abortall = True
        self.abortSingleClick()

        self.abortall = False
        self.scanclick = False
        self.pending = False
        self.coordsReady = 0
        self.motorReady = 0
        self.pause = False
        self.abortall_btn['state'] = tk.DISABLED
        self.scan_btn['state'] = tk.NORMAL
        self.pause_btn['state'] = tk.DISABLED
    
    def update_detCheckMsg(self, *args):
        self.detCheck_msg.set('Det Check (sec) = %s'%(self.detCheck_val.get()))
        
    def update_detCheckValState(self):
        if self.detectorMonitor.get():
            self.detCheck_entry['state'] = tk.DISABLED
        else:
            self.detCheck_entry['state'] = tk.NORMAL

    
    def __init__(self, tabControl, setup_tab):
        self.scanfrm = ttk.Frame(tabControl)
        self.inputs_labels = setup_tab.inputs_labels
        self.calctime_out = setup_tab.calctime_out
        self.scanType = setup_tab.scanType
        self.smp_name = setup_tab.smp_name
        self.bda = setup_tab.bda
        self.tot_time = setup_tab.tot_time
        self.ptycho = setup_tab.ptychoVal
        self.scanParms = setup_tab.scanParms
        self.scan_start_time = 0
        self.eta_str = ''
        self.cline = 0
        self.det_reset_attemp = 0
        self.cline_time = 0
        self.single_line_time = 0
        self.slist = scanList(self.scanfrm, self.inputs_labels, self.calctime_out,
                 self.scanType, self.smp_name, self.bda, self.tot_time, self.ptycho, self.scanParms)
        self.insertScan = self.slist.insertScan

        self.scanclick = False
        self.pause = False
        self.abortsingle = False
        self.abortall = False
        self.pending = False
        self.scanning = False
        self.ycenter_check = False
        self.detector_resetting = False
        self.eigerReady = False
        
        self.coordsReady = 0
        self.coarse_scnum = ''
        self.motorReady = 0
        self.scdic = None
        self.stype = 'XRF'
        self.motors = []
        self.scandone_var = tk.BooleanVar()
        self.scandone_var.set(False)
        self.scandone_var.trace('w', self.scanDone)
        self.record = None
        self.recordval = None
        self.parm = {}
        self.pvComm = pvComm()
        
        self.pbarscmsg = tk.StringVar()
        self.pbarscmsg.set('%s'%(self.pvComm.nextScanName()))
        pbar_sc_txt = tk.Label(self.scanfrm, textvariable = self.pbarscmsg)
        pbar_sc_txt.grid(row = 23, column = 5, sticky='w', pady = (35, 0),
                             padx = (20,0))
        self.pbarscval = tk.DoubleVar()
        self.pbarscval.set(0.0)
        self.pbar_sc = ttk.Progressbar(self.scanfrm, orient = tk.HORIZONTAL, 
                                           length = 200, mode = 'determinate',
                                           variable = self.pbarscval)
        self.pbar_sc.grid(row = 23, column = 5, columnspan = 3, sticky='w', 
                              pady = (35,0), padx=(290,0))
        
        self.monitormsg = tk.StringVar()
        self.monitormsg.set('Not scanning, not active')
        self.mmsg_label = tk.Label(self.scanfrm, textvariable = self.monitormsg)
        self.mmsg_label.grid(row = 24, column = 1, columnspan = 5, padx=(20,0),
                             sticky = 'w')
        
        self.detectorMonitor = tk.IntVar()
        #self.detectorMonitor.set(0)
        self.detectorMonitor.set(1)
        detMonitor_btn = ttk.Checkbutton(
            master=self.scanfrm, text='Detector Monitor', variable=self.detectorMonitor, 
            command=self.update_detCheckValState)
        detMonitor_btn.grid(row=24, column=4)
        
        self.detCheck_msg = tk.StringVar()
        detCheck_txt = tk.Label(self.scanfrm, 
                                textvariable=self.detCheck_msg)
        detCheck_txt.grid(row = 24, column = 5, sticky='w')
        
        vcmd = self.scanfrm.register(checkEntryDigit)
        self.detCheck_val = tk.StringVar()
        self.detCheck_val.trace('w', self.update_detCheckMsg)
        self.detCheck_val.set('%d'%20)
        self.detCheck_entry = tk.Entry(self.scanfrm, width=5, textvariable=self.detCheck_val,
                                  validate='all', validatecommand=(vcmd, "%P"))
        self.detCheck_entry.grid(row = 24, column=5, sticky='w', padx=(150,0))
        self.detCheck_entry['state'] = tk.DISABLED

     
        self.scmsg = tk.Text(self.scanfrm, wrap = 'word', height = 15, width = 150)
        self.scmsg.grid(row = 24, column = 1, sticky = 'w', columnspan = 10, 
                        rowspan = 8, padx=(20,0), pady=(5,0))
        stdoutToTextbox(self.scmsg)
        
        row = 23
        clearsclist_btn = tk.Button(self.scanfrm, text = 'Clear all', command = self.slist.clearSclist, width = 20)
        clearsclist_btn.grid(row=row, column=0, sticky='w', pady=(15,0), padx=(20,0))
        
        rmselect_btn = tk.Button(self.scanfrm, text = 'Remove selected', command = self.slist.removeSelect, width = 20)
        rmselect_btn.grid(row=row+1, column=0, sticky = 'w', pady=(15,0), padx=(20,0))
        
        self.scan_btn = tk.Button(self.scanfrm, text = 'Scan', command = self.scanClick, width = 20)
        self.scan_btn.grid(row=row+2, column=0, sticky = 'w', pady = (15, 0), padx=(20,0))
        
        self.pause_btn = tk.Button(self.scanfrm, text = 'Pause', command = self.pauseClick, width = 20)
        self.pause_btn.grid(row=row+3, column=0, sticky = 'w', pady = (15, 0), padx=(20,0))
        self.pause_btn['state'] = tk.DISABLED
        
        self.resume_btn = tk.Button(self.scanfrm, text = 'Resume', command = self.resumeClick, width = 20)
        self.resume_btn.grid(row =row+4, column = 0, sticky = 'w', pady = (15, 0), padx=(20,0))
        self.resume_btn['state'] = tk.DISABLED
        
        self.abort_btn = tk.Button(self.scanfrm, text = 'Abort Single Scan', command = self.abortSingleClick, width = 20)
        self.abort_btn.grid(row =row+5, column = 0, sticky = 'w', pady = (15, 0), padx=(20,0))
        self.abort_btn['state'] = tk.DISABLED
        
        self.abortall_btn = tk.Button(self.scanfrm, text = 'Abort Batch', command = self.abortAllClick, width = 20)
        self.abortall_btn.grid(row =row+6, column = 0, sticky = 'w', pady = (15, 0), padx=(20,0))
        self.abortall_btn['state'] = tk.DISABLED
        
        self.logTemp = tk.IntVar()
        logTemp_chckbx = tk.Checkbutton(self.scanfrm, text = 'Log Temperature',
                                        variable = self.logTemp, width = 20)
        logTemp_chckbx.grid(row =24, column = 5, padx=(200,0))
        #----------------------------------for test imges-xyl----------------------------------------------------
        self.x_scan_list = []
        self.y_scan_list = []
        self.angle_list = []
        self.real_ang_test = []
        
        #-------------------------xyl: cur directory------------------
        self.rootfolder = PV.caget('9idbBNP:saveData_fileSystem')
        self.rootfolder = self.rootfolder.replace('//micdata/data1','/mnt/micdata1')
        self.user = PV.caget('9idbBNP:saveData_subDir').split('/')[0]
        self.user_folder = os.path.join(self.rootfolder,self.user)

        #-----------------------------------------------------------------------------------------------------------
        try:
            self.mmsg_label.after(1000, self.scanMonitor)
        except:
            pass

