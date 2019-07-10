import struct
import numpy as np


from biologicalgraphs.utilities.constants import *



class Joint:
    def __init__(self, iv, iz, iy, ix):
        self.iv = iv
        self.iz = iz
        self.iy = iy
        self.ix = ix



class Endpoint:
    def __init__(self, iv, iz, iy, ix, vector):
        self.iv = iv
        self.iz = iz
        self.iy = iy
        self.ix = ix
        self.vector = np.array(vector, dtype=np.float32)


class Skeleton:
    def __init__(self, label, joints, endpoints, vectors, resolution, grid_size):
        self.label = label
        self.grid_size = grid_size
        self.resolution = resolution
        self.joints = []
        self.endpoints = []        

        for joint in joints:
            iz = joint / (grid_size[IB_Y] * grid_size[IB_X])
            iy = (joint - iz * grid_size[IB_Y] * grid_size[IB_X]) / grid_size[IB_X]
            ix = joint % grid_size[IB_X]

            self.joints.append(Joint(joint, iz, iy, ix))

        for endpoint in endpoints:
            iz = endpoint / (grid_size[IB_Y] * grid_size[IB_X])
            iy = (endpoint - iz * grid_size[IB_Y] * grid_size[IB_X]) / grid_size[IB_X]
            ix = endpoint % grid_size[IB_X]

            vector = vectors[endpoint]

            self.endpoints.append(Endpoint(endpoint, iz, iy, ix, vector))