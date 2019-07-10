import h5py
import struct


import numpy as np


from biologicalgraphs.data_structures import meta_data, skeleton_points
from biologicalgraphs.utilities.constants import *



def GridSize(prefix):
    # return the size of this dataset
    return meta_data.MetaData(prefix).GridSize()



def CroppingBox(prefix):
    # return which locations are valid for training and validation
    return meta_data.MetaData(prefix).CroppingBox()



def ReadMetaData(prefix):
    # return the meta data for this prefix
    return meta_data.MetaData(prefix)



def Resolution(prefix):
    # return the resolution for this prefix
    return meta_data.MetaData(prefix).Resolution()



def GetGoldFilename(prefix):
    filename, _ = meta_data.MetaData(prefix).GoldFilename()

    return filename



def ReadH5File(filename, dataset=None):
    # read the h5py file
    with h5py.File(filename, 'r') as hf:
        # read the first dataset if none given
        if dataset == None: data = np.array(hf[hf.keys()[0]])
        else: data = np.array(hf[dataset])

        # allow affinities and images to not be int64, everything else gets converted
        if data.dtype == np.float32 or data.dtype == np.uint8 or data.dtype == np.int64: return data
        else: return data.astype(np.int64)



def WriteH5File(data, filename, dataset, compression=True):
    with h5py.File(filename, 'w') as hf:
        # should cover all cases of affinities/images
        if compression: hf.create_dataset(dataset, data=data, compression='gzip')
        else: hf.create_dataset(dataset, data=data)



def ReadAffinityData(prefix):
    filename, dataset = meta_data.MetaData(prefix).AffinityFilename()

    affinities = ReadH5File(filename, dataset).astype(np.float32)

    # create the dataset so it is (z, y, x, c)
    if affinities.shape[0] == 3: affinities = np.moveaxis(affinities, 0, 3)

    return affinities



def ReadGoldData(prefix):
    filename, dataset = meta_data.MetaData(prefix).GoldFilename()

    return ReadH5File(filename, dataset).astype(np.int64)



def ReadImageData(prefix):
    filename, dataset = meta_data.MetaData(prefix).ImageFilename()

    return ReadH5File(filename, dataset)



def ReadSegmentationData(prefix):
    filename, dataset = meta_data.MetaData(prefix).SegmentationFilename()

    return ReadH5File(filename, dataset).astype(np.int64)



def ReadSkeletons(prefix):
    # parameters for low resolution segmentations
    downsample_resolution = (80, 80, 80)

    # read in all of the skeleton points
    skeleton_filename = 'skeletons/{}/thinning-{:03d}x{:03d}x{:03d}-upsample-skeleton.pts'.format(prefix, downsample_resolution[IB_X], downsample_resolution[IB_Y], downsample_resolution[IB_Z])
    endpoint_filename = 'skeletons/{}/thinning-{:03d}x{:03d}x{:03d}-endpoint-vectors.vec'.format(prefix, downsample_resolution[IB_X], downsample_resolution[IB_Y], downsample_resolution[IB_Z])

    # read the joints file and the vector file
    with open(skeleton_filename, 'rb') as sfd, open(endpoint_filename, 'rb') as efd:
        skel_zres, skel_yres, skel_xres, skel_max_label, = struct.unpack('qqqq', sfd.read(32))
        end_zres, end_yres, end_xres, end_max_label, = struct.unpack('qqqq', efd.read(32))
        assert (skel_zres == end_zres and skel_yres == end_yres and skel_xres == end_xres and skel_max_label == end_max_label)    

        # create an array of skeletons
        skeletons = []
        resolution = Resolution(prefix)
        grid_size = GridSize(prefix)

        for label in range(skel_max_label):
            joints = []
            endpoints = []
            vectors = {}

            nelements, = struct.unpack('q', sfd.read(8))
            for _ in range(nelements):
                index, = struct.unpack('q', sfd.read(8))
                if (index < 0): endpoints.append(-1 * index)
                else: joints.append(index)

            nendpoints, = struct.unpack('q', efd.read(8))
            assert (len(endpoints) == nendpoints)
            for _ in range(nendpoints):
                endpoint, vz, vy, vx, = struct.unpack('qddd', efd.read(32))

                vectors[endpoint] = (vz, vy, vx)

            skeletons.append(skeleton_points.Skeleton(label, joints, endpoints, vectors, resolution, grid_size))

    return skeletons



def SpawnMetaFile(prefix, segment_filename, segment_dataset):
    meta = meta_data.MetaData(prefix)

    # get the new prefix for the data from the segment file
    new_prefix = segment_filename.split('/')[1][:-3]

    # update the values for this meta data
    meta.prefix = new_prefix
    meta.segment_filename = '{} {}'.format(segment_filename, segment_dataset)

    meta.WriteMetaFile()
