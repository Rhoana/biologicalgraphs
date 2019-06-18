# general functions for transforming h5 files
from ibex.utilities.constants import *
from numba import jit
import numpy as np
import math


# downsample the data by (z, y, x) ratio
@jit(nopython=True)
def DownsampleData(data, ratio=(1, 2, 2)):
    # get the size of the current dataset
    (zres, yres, xres) = data.shape

    # create an empty array for downsampling
    (down_zres, down_yres, down_xres) = (int(zres / ratio[IB_Z]), int(yres / ratio[IB_Y]), int(xres / ratio[IB_X]))
    downsampled_data = np.zeros((down_zres, down_yres, down_xres), dtype=data.dtype)
    
    # fill in the entries of the array
    for iz in range(down_zres):
        for iy in range(down_yres):
            for ix in range(down_xres):
                downsampled_data[iz,iy,ix] = data[int(iz * ratio[IB_Z]), int(iy * ratio[IB_Y]), int(ix * ratio[IB_X])]

    return downsampled_data



@jit(nopython=True)
def MaskAndCropSegmentation(data, labels):
    # create a set of valid segments
    ids = set()
    for label in labels:
        ids.add(label)

    # get the shape of the data
    zres, yres, xres = data.shape

    zmin, ymin, xmin = data.shape
    zmax, ymax, xmax = (0, 0, 0)

    masked_data = np.zeros((zres, yres, xres), dtype=np.int64)

    # go through the entire data set
    for iz in range(zres):
        for iy in range(yres):
            for ix in range(xres):
                # skip masked out values
                if not data[iz,iy,ix] in ids: continue
                
                masked_data[iz,iy,ix] = data[iz,iy,ix]
                if iz < zmin: zmin = iz
                if iy < ymin: ymin = iy
                if ix < xmin: xmin = ix
                if iz > zmax: zmax = iz
                if iy > ymax: ymax = iy
                if ix > xmax: xmax = ix

    return masked_data[zmin:zmax,ymin:ymax,xmin:xmax]



# split the data to create training and validation data
@jit(nopython=True)
def SplitData(data, axis, threshold=0.5):
    assert (0 <= axis and axis <= 2)

    # get the separation index
    separation = int(threshold * data.shape[axis])

    # split the data into two components
    if (axis == 0):
        training_data = data[0:separation,:,:]
        validation_data = data[separation:,:,:]
    elif (axis == 1):
        training_data = data[:,0:separation,:]
        validation_data = data[:,separation:,:]
    else:
        training_data = data[:,:,0:separation]
        validation_data = data[:,:,separation:]
        
    # return the training and validation data
    return training_data, validation_data