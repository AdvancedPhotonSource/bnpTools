# updates:
bnp_gui is used in bionanoprobe beamline for data collection, originally developed by Grace Luo. I worked to resolve some issues and improve the gui (I should have comments and lines (------, ----xyl---, etc.) or flags to specify the parts that I wrote). My changes are in bnp_gui, imgprocessing, pvComm, scanBNP, scanFrame, setupFrame, setupScan.py:

For XRF tomography data collection: the data collection stops when k-mean cluster segmentation did not work. I changed to save coarse images, histogram and fine images to the user folder. I also saved the sample xyz position as an csv file for users to check position later. I also added filters in k-mean cluster segmentation.
I added user folder selection option instead of manually typing and creating user folders (flyXRF, mda etc.).
I added XANES data collection workflow
I added function to show ptychography image in the GUI, so users can select regions in ptychography image
Add small but convinient functions such as export images, records xy coordinates, etc. for users and revise the GUI interface
