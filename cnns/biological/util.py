import random
import numpy as np
import scipy

from numba import jit

from ibex.utilities.constants import *



@jit(nopython=True)
def GenerateExampleFromSegment(segment, width, indices):
    nchannels, zres, yres, xres = width

    example = np.zeros((nchannels, zres, yres, xres), dtype=np.float32)

    # go through every element and set the appropriate channel values
    for iz in range(zres):
        for iy in range(yres):
            for ix in range(xres):
                if nchannels == 1:
                    if segment[iz,iy,ix]:
                        example[0,iz,iy,ix] = 1
                else:
                    # use indices array to avoid learning label one and label two differences
                    if segment[iz,iy,ix] == 1:
                        example[0,iz,iy,ix] = 1
                        example[indices[0],iz,iy,ix] = 1
                    elif segment[iz,iy,ix] == 2:
                        example[0,iz,iy,ix] = 1
                        example[indices[1],iz,iy,ix] = 1

    # subtract 0.5 so that the values are {-0.5, 0.5}
    return example - 0.5



def AugmentFeature(segment, width):
    # to randomize the first and second channels (ordering of label one and label two)
    # need this outside of GenerateExampleFromSegment for @jit(nopython) to work
    indices = [1, 2]
    random.shuffle(indices)

    # generate the example from the segment by creating nchannels binary channels
    example = GenerateExampleFromSegment(segment, width, indices)

    # expand the first dimension
    example = np.expand_dims(example, 0)

    # randomly flip the angle
    if random.random() > 0.5: example = np.flip(example, IB_Z + 2)

    # randomly rotate the example
    angle = random.uniform(0, 360)
    example = scipy.ndimage.interpolation.rotate(example, angle, axes=(IB_X + 2, IB_Y + 2), reshape=False, order=0, mode='constant', cval=-0.5)

    return example