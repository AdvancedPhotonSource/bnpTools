#!/APSshare/anaconda3/x86_64/bin/python
'''
This is for X-piezo scan (stage coordinates) and Y-combined (stage coordinates) batch flyscan with different sample_Z postions.

The max scan width in X direction is 80 um.
'''

import epics
from epics import caput, caget
from epics import PV
import time
import numpy as np
import pdb
import tqdm

'''
please enter the scan prameters below:
scans [x-center(um), y-center.(um), z-position (um), x-width.(um), y-width.(um), x-stepsize.(um), Y-stepsize.(um), dwell.(ms)]
'''

#caput('9idbTAU:SM:Ps:xyDiffMotion.VAL', 1) #0: off; 1: on

scans = [




]










#caput('9idbTAU:SM:ST:RqsPos',tilt_angle)
#BDA IN
#BDA_IN=-8517 #-8607 #-7992.7 #-8607
#caput('21:D3:UA:UX:RqsPos',BDA_IN)

                 
pvs = ['9idbTAU:SM:PX:RqsPos', '9idbTAU:SY:PY:RqsPos', '9idbTAU:SM:SZ:RqsPos', 
       '9idbBNP:scan1.P1WD', '9idbBNP:scan2.P1WD', '9idbBNP:scan1.P1SI', 
       '9idbBNP:scan2.P1SI', '9idbBNP:scanTran3.C PP']

sm_px_RqsPos=PV('9idbTAU:SM:PX:RqsPos')
sm_px_ActPos=PV('9idbTAU:SM:PX:ActPos') 
sm_py_RqsPos=PV('9idbTAU:SY:PY:RqsPos') 
sm_py_ActPos=PV('9idbTAU:SY:PY:ActPos')

# Prepare tqdm outer progress bar
print('Batchscan starts')

outer = tqdm.tqdm(total = len(scans), desc = 'Number of Scans', position = 0, ncols=100)

for batch_num, scan in enumerate(scans):
        #pdb.set_trace()
	outer.write('Changing XY scan mode to combined motion.')
	caput('9idbTAU:SM:Ps:xMotionChoice.VAL', 0)  #0: Stepper+piezo, 1: stepper only, 2: piezo only
	time.sleep(2.)
	caput('9idbTAU:SY:Ps:yMotionChoice.VAL', 0)
	time.sleep(2.)

	#print 'scan #{0:d} starts'.format(batch_num)
	outer.write('Entering scan parameters for scan %d.'%(batch_num+1))
	for i, pvs1 in enumerate(pvs):
		#print 'Setting %s' %pvs1 
		caput(pvs1, scans[batch_num][i])
		time.sleep(0.2)
 
        #check whether the motors have moved to the requested position 
	outer.write('Checking whether motors are in position.')
	ready=abs(sm_px_ActPos.get()-sm_px_RqsPos.get())<0.05 and abs(sm_py_ActPos.get()-sm_py_RqsPos.get())<0.05
	while not ready:
		outer.write('\t Motors are not ready, sad...')
		sm_px_RqsPos.put(sm_px_RqsPos.get())
		sm_py_RqsPos.put(sm_py_RqsPos.get())
		time.sleep(3.)
		ready=abs(sm_px_ActPos.get()-sm_px_RqsPos.get())<0.05 and abs(sm_py_ActPos.get()-sm_py_RqsPos.get())<0.05
	outer.write('\t Motors are ready now.')
	outer.write('Setting the current position as the center of the scan.')

	caput('9idbBNP:aoRecord11.PROC', 1)
	time.sleep(1.)
	caput('9idbBNP:aoRecord12.PROC', 1)
	time.sleep(1.)
	
	outer.write('Changing X scan mode to Piezo only.')
	caput('9idbTAU:SM:Ps:xMotionChoice.VAL', 2)
	time.sleep(1.)

	outer.write('Centering piezoX and piezoY.')
	caput('9idbTAU:SM:Ps:xCenter.PROC', 1)
	time.sleep(3.)
	caput('9idbTAU:SY:Ps:yCenter.PROC', 1)
	time.sleep(3.)
	caput('9idbTAU:SM:Ps:xCenter.PROC', 1)
	time.sleep(3.)
	caput('9idbTAU:SY:Ps:yCenter.PROC', 1)
	time.sleep(3.)
	
        #BDA in
        #caput('21:D3:UA:UX:RqsPos',BDA_IN)
	caput('9idbBNP:scan2.EXSC', 1)
	time.sleep(1.)
	done = False
	outer.write ('Checking every 10 sec for scan to complete.')
	
	tlines = caget('9idbBNP:scan2.NPTS')
	cline = 0
	inner = tqdm.tqdm(total=tlines, desc='Scan %d'%(batch_num+1), position=batch_num+1, ascii=True, ncols=80)
	while not done:
		done = (caget('9idbBNP:scan2.EXSC')==0)
		cline = caget('9idbBNP:scan2.CPT')
		inner.update(cline-inner.n)
		#print('\t Batch %d/%d scan is ongoing'%(batch_num+1, len(scans)))
		#inner.write('\t Scan %d/%d is going, progress: %d/%d lines.'%(batch_num+1, len(scans),cline, tlines))
		time.sleep(5.)
	inner.close()
	outer.update(1)

        #BDA out
        #caput('21:D3:UA:UX:RqsPos',BDA_IN-500)

outer.close()

print('%sCompleted. Congratulations!'%('\n'*(batch_num+1)))
caput('9idbTAU:SM:Ps:xMotionChoice.VAL', 0)
#caput('9idbTAU:SM:Ps:xyDiffMotion.VAL', 0)
#BDA out
#caput('21:D3:UA:UX:RqsPos',BDA_IN-500)
#caput('9ida:rShtrB:Close', 1)
input("\n\nPress Return to exit")


