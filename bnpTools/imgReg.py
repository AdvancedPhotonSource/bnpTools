#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os, sys, cv2
sys.path.append('/Users/yluo89/Library/CloudStorage/Box-Box/9-BNP')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure
from matplotlib.patches import Rectangle
from bnp_tools.analysis import four_point_transform, getNorm8U
import collections
from skimage.transform import rescale


# In[3]:


#%% Feature matching
def siftFeatureDetect(im1,im2, sigma = None, nOct = None, im1mask=None, 
                      im2mask = None, limit = None, axes = None, plotFig = True):
    sift = cv2.SIFT_create(sigma=sigma, nOctaveLayers = nOct)
    kp1, des1 = sift.detectAndCompute(im1,im1mask)
    kp2, des2 = sift.detectAndCompute(im2,im2mask)
    
#     print(des1.dtype, des2.dtype)
#     if type(des1) != CV_32F:
#         des1 = np.float32(des1)
#     if type(des2) != CV_32F:
#         des2 = np.float32(des2)
#     print(des1.dtype, des2.dtype)
    
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 500)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    im2nolines = im2.copy()
        
    # store all the good matches as per Lowe's ratio test.
    good = []
    if limit is None:
        limit = 0.7
    for m,n in matches:
        if m.distance < limit*n.distance:
            good.append(m)
    
    MIN_MATCH_COUNT = 3
    dst = None
    M = None
    if len(good)>=MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    
        M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, M, int(cv2.RANSAC), 5.0, 
                                     maxIters = 5000, confidence=0.99)
        matchesMask = mask.ravel().tolist()
    
        h,w = im1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.transform(pts, M)
#         dst = cv2.affineTransform(pts, M)
#         dst = cv2.perspectiveTransform(pts,M)
    
        im2 = cv2.polylines(im2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    
    else:
        print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
        matchesMask = None
        
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                        singlePointColor = None,
                        matchesMask = matchesMask, # draw only inliers
                        flags = 2)
    
    img3 = cv2.drawMatches(im1,kp1,im2,kp2,good,None,**draw_params)
    # img3 = cv2.drawMatchesKnn(im1,kp1,im2,kp2,good,outImg=None,flags=2)
    img2 = cv2.drawKeypoints(im2nolines, kp2, None, color = (0,255,0))
    img1 = cv2.drawKeypoints(im1.copy(), kp1, None, color = (0,255,0))
    
    if axes is not None:
        axes.imshow(img3, 'gray')
        axes.axis('off')
    if plotFig:
        plt.imshow(img3,'gray')
        plt.axis('off')

    return {'combined':img3, 'kpts_img2':img2, 'kpts_img1':img1, 'M':M, 'dstPts':dst,
           'im2_box':im2, 'good':good}

def imgRegHNSCC(refmap, movmap, movscale = 1, siftLimit = None, nOct = None, sigma=None, plotFig=False):
    refmap = exposure.equalize_hist(refmap)
    movmap = exposure.equalize_hist(movmap)
    movmap1 = rescale(movmap, movscale, anti_aliasing=True)
    movmap1 = exposure.match_histograms(movmap1, refmap)
    movmap1 = exposure.rescale_intensity(movmap1)
    a = siftFeatureDetect(getNorm8U(movmap1), getNorm8U(refmap), limit = siftLimit, 
                          nOct = nOct, sigma = sigma, plotFig = plotFig)
    
    if a['dstPts'] is not None:
        dstr = ['movmap1', 'refreg', 'refbox']
        dlabel = ['Img1', 'Img2_cropped', 'Img2']
        refreg = four_point_transform(refmap, a['dstPts'].squeeze())
        if np.nanstd(refreg) > 0.1:
            fig,axes = plt.subplots(3,1,figsize=(3,9))
            refbox = a['im2_box']
            for s_, l_, ax_ in zip(dstr, dlabel, axes.ravel()):
                d = eval(s_)
                if 'reg' in s_:
                    vmax = None
                else:
                    vmax = np.mean(d) + 1.5*np.std(d)
                ax_.imshow(exposure.equalize_hist(d), cmap = 'gray')
                ax_.set_xticks([])
                ax_.set_yticks([])
                ax_.axis('scaled')
                ax_.set_title(l_)

            return fig


# In[ ]:




