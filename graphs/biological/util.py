import os
import numpy as np

from numba import jit

from biologicalgraphs.utilities.constants import *
from biologicalgraphs.utilities import dataIO


@jit(nopython=True)
def FindSmallSegments(segmentation, threshold):
    # create lists for small and large nodes
    small_segments = set()
    large_segments = set()

    zres, yres, xres = segmentation.shape

    # create a count for each label
    max_label = np.amax(segmentation) + 1
    counts = np.zeros(max_label, dtype=np.int64)

    for iz in range(zres):
        for iy in range(yres):
            for ix in range(xres):
                counts[segmentation[iz,iy,ix]] += 1

    for label in range(max_label):
        if not counts[label]: continue

        if (counts[label] < threshold): small_segments.add(label)
        else: large_segments.add(label)

    return small_segments, large_segments




@jit(nopython=True)
def ScaleFeature(segment, width, label_one, label_two):
    # get the size of the extracted segment
    zres, yres, xres = segment.shape

    example = np.zeros((width[IB_Z], width[IB_Y], width[IB_X]), dtype=np.uint8)

    # iterate over the example coordinates
    for iz in range(width[IB_Z]):
        for iy in range(width[IB_Y]):
            for ix in range(width[IB_X]):
                # get the global coordiantes from segment
                iw = int(float(zres) / float(width[IB_Z]) * iz)
                iv = int(float(yres) / float(width[IB_Y]) * iy)
                iu = int(float(xres) / float(width[IB_X]) * ix)

                if segment[iw,iv,iu] == label_one:
                    example[iz,iy,ix] = 1
                elif segment[iw,iv,iu] == label_two:
                    example[iz,iy,ix] = 2

    return example




@jit(nopython=True)
def ExtractExample(segment, label_one, label_two):
    zres, yres, xres = segment.shape

    for iz in range(zres):
        for iy in range(yres):
            for ix in range(xres):
                if (not segment[iz,iy,ix] == label_one) and (not segment[iz,iy,ix] == label_two):
                    segment[iz,iy,ix] = 0

    return segment



# simple function to create directory structure for all of the features
def CreateDirectoryStructure(width, network_radius, subsets, feature):
    if not os.path.exists('features'):
        os.mkdir('features')
    if not os.path.exists('features/biological'):
        os.mkdir('features/biological')

    # make sure directory structure exists
    directory = 'features/biological/{}-{}nm-{}x{}x{}'.format(feature, network_radius, width[IB_Z], width[IB_Y], width[IB_X])
    if not os.path.exists(directory):
        os.mkdir(directory)

    # add all subsets
    for subset in subsets:
        sub_directory = '{}/{}'.format(directory, subset)
        if not os.path.exists(sub_directory):
            os.mkdir(sub_directory)
        # there are three possible labels per subset
        labelings = ['positives', 'negatives', 'unknowns']
        for labeling in labelings:
            if not os.path.exists('{}/{}'.format(sub_directory, labeling)):
                os.mkdir('{}/{}'.format(sub_directory, labeling))



def GenerateExamplesArray(prefix, segmentation, examples, width, network_radius):
    # get the radius along each dimensions in terms of voxels
    resolution = dataIO.Resolution(prefix)
    (zradius, yradius, xradius) = (int(network_radius / resolution[IB_Z]), int(network_radius / resolution[IB_Y]), int(network_radius / resolution[IB_X]))
    zres, yres, xres = segmentation.shape

    # find the number of examples
    nexamples = len(examples)

    # create the empty array of examples
    examples_array = np.zeros((nexamples, width[IB_Z], width[IB_Y], width[IB_X]), dtype=np.uint8)

    for index, (zpoint, ypoint, xpoint, label_one, label_two) in enumerate(examples):
        # need to make sure that bounding box does not leave location so sizes are correct
        zmin = max(0, zpoint - zradius)
        ymin = max(0, ypoint - yradius)
        xmin = max(0, xpoint - xradius)
        zmax = min(zres, zpoint + zradius + 1)
        ymax = min(yres, ypoint + yradius + 1)
        xmax = min(xres, xpoint + xradius + 1)

        # create the empty example file with three channels corresponding to the value of segment
        example = np.zeros((2 * zradius + 1, 2 * yradius + 1, 2 * xradius + 1), dtype=np.int32)

        # get the valid location around this point
        segment = ExtractExample(segmentation[zmin:zmax,ymin:ymax,xmin:xmax].copy(), label_one, label_two)

        if example.shape == segment.shape:
            example = segment
        else:
            if zmin == 0: zstart = zradius - zpoint
            else: zstart = 0

            if ymin == 0: ystart = yradius - ypoint
            else: ystart = 0

            if xmin == 0: xstart = xradius - xpoint
            else: xstart = 0

            # the second and third channels are one if the corresponding voxels belong to the individual segments
            example[zstart:zstart+segment.shape[IB_Z],ystart:ystart+segment.shape[IB_Y],xstart:xstart+segment.shape[IB_X]] = segment

        # scale the feature to the appropriate width
        examples_array[index,:,:,:] = ScaleFeature(example, width, label_one, label_two)

    # return the examples
    return examples_array
