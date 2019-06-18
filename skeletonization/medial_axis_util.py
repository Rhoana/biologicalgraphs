from numba import jit



# function takes the skeleton from the medial axis algorithm and returns
# a list of joints with negative values indicating endpoints
@jit(nopython=True)
def PostProcess(skeleton):
    zres, yres, xres = skeleton.shape

    # keep track of all the endpoints
    joints = []

    for iz in range(zres):
        for iy in range(yres):
            for ix in range(xres):
                if not skeleton[iz,iy,ix]: continue

                linear_index = iz * yres * xres + iy * xres + ix

                # find out how many element in the local neighborhood are skeleton
                nneighbors = 0
                for iw in range(iz - 1, iz + 2):
                    if iw < 0 or iw > zres - 1: continue
                    for iv in range(iy - 1, iy + 2):
                        if iv < 0 or iv > yres - 1: continue
                        for iu in range(ix - 1, ix + 2):
                            if iu < 0 or iu > xres - 1: continue

                            if skeleton[iw,iv,iu]: nneighbors += 1

                if nneighbors <= 2: joints.append(-1 * linear_index)
                else: joints.append(linear_index)
    
    # return the list of all the endpoint
    return joints
