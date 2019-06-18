import struct
import numpy as np


from ibex.utilities.constants import *



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




# class Skeleton:
#     def __init__(self, label, joints, endpoints):
#         self.label = label
#         self.joints = joints
#         self.endpoints = endpoints

#     def NPoints(self):
#         return len(self.joints) + len(self.endpoints)

#     def NEndpoints(self):
#         return len(self.endpoints)

#     def NJoints(self):
#         return len(self.joints)

#     def Endpoints2Array(self):
#         nendpoints = len(self.endpoints)

#         array = np.zeros((nendpoints, 3), dtype=np.int64)
#         for ie in range(nendpoints):
#             array[ie] = self.endpoints[ie]

#         return array

#     def Joints2Array(self):
#         njoints = len(self.endpoints) + len(self.joints)

#         array = np.zeros((njoints, 3), dtype=np.int64)
#         index = 0
#         for endpoint in self.endpoints:
#             array[index] = endpoint
#             index += 1
#         for joint in self.joints:
#             array[index] = joint
#             index += 1

#         return array


#     def WorldJoints2Array(self, resolution):
#         njoints = len(self.endpoints) + len(self.joints)

#         array = np.zeros((njoints, 3), dtype=np.int64)
#         index = 0
#         for endpoint in self.endpoints:
#             array[index] = (endpoint[IB_Z] * resolution[IB_Z], endpoint[IB_Y] * resolution[IB_Y], endpoint[IB_X] * resolution[IB_X])
#             index += 1
#         for joint in self.joints:
#             array[index] = (joint[IB_Z] * resolution[IB_Z], joint[IB_Y] * resolution[IB_Y], joint[IB_X] * resolution[IB_X])
#             index += 1

#         return array


# class Skeletons:
#     def __init__(self, prefix, skeleton_algorithm, downsample_resolution, benchmark, params):
#         self.skeletons = []

#         # read in all of the skeleton points
#         if benchmark: filename = 'benchmarks/skeleton/{}-{}-{:03d}x{:03d}x{:03d}-upsample-{}-skeleton.pts'.format(prefix, skeleton_algorithm, downsample_resolution[IB_X], downsample_resolution[IB_Y], downsample_resolution[IB_Z], params)
#         else: filename = 'skeletons/{}/{}-{:03d}x{:03d}x{:03d}-upsample-{}-skeleton.pts'.format(prefix, skeleton_algorithm, downsample_resolution[IB_X], downsample_resolution[IB_Y], downsample_resolution[IB_Z], params)

#         with open(filename, 'rb') as fd:
#             zres, yres, xres, max_label, = struct.unpack('qqqq', fd.read(32))

#             for label in range(max_label):
#                 joints = []
#                 endpoints = []

#                 njoints, = struct.unpack('q', fd.read(8))
#                 for _ in range(njoints):
#                     iv, = struct.unpack('q', fd.read(8))
                    
#                     # endpoints are negative
#                     endpoint = False
#                     if (iv < 0): 
#                         iv = -1 * iv 
#                         endpoint = True

#                     iz = iv / (yres * xres)
#                     iy = (iv - iz * yres * xres) / xres
#                     ix = iv % xres

#                     if endpoint: endpoints.append((iz, iy, ix))
#                     else: joints.append((iz, iy, ix))

#                 self.skeletons.append(Skeleton(label, joints, endpoints))

#     def NSkeletons(self):
#         return len(self.skeletons)


#     def KthSkeleton(self, k):
#         return self.skeletons[k]