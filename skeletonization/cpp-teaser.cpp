/* c++ file for the teaser skeletonization strategy */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "cpp-MinBinaryHeap.h"
#include "cpp-generate_skeletons.h"



// global variables
static long grid_size[3];
static long nentries;
static long sheet_size;
static long row_size;
static long infinity;



static unsigned char *segmentation = NULL;
static unsigned char *skeleton = NULL;
static double *DBF = NULL;
static double *penalties = NULL;
static double *PDRF = NULL;
static unsigned char *inside = NULL;



static long inside_voxels = 0;
static double scale = 1.3;
static long buffer = 2;
static const long min_path_length = 2;



static void IndexToIndicies(long iv, long &ix, long &iy, long &iz)
{
    iz = iv / sheet_size;
    iy = (iv - iz * sheet_size) / row_size;
    ix = iv % row_size;
}



static long IndicesToIndex(long ix, long iy, long iz)
{
    return iz * sheet_size + iy * row_size + ix;
}



static void ComputeDistanceFromBoundaryField(void)
{
    // allocate memory for bounday map and distance transform
    long *b = new long[nentries];
    for (long iz = 0; iz < grid_size[IB_Z]; ++iz) {
        for (long iy = 0; iy < grid_size[IB_Y]; ++iy) {
            for (long ix = 0; ix < grid_size[IB_X]; ++ix) {
                if (!segmentation[IndicesToIndex(ix, iy, iz)]) {
                    b[IndicesToIndex(ix, iy, iz)] = 0;
                    continue;
                }

                // inside voxels that are on the boundary have value 1 (based on TEASER paper figure 2)
                if ((ix == 0 or iy == 0 or iz == 0 or (ix == grid_size[IB_X] - 1) or (iy == grid_size[IB_Y] - 1) or (iz == grid_size[IB_Z] - 1)) ||
                    (ix > 0 and !segmentation[IndicesToIndex(ix - 1, iy, iz)]) ||
                    (iy > 0 and !segmentation[IndicesToIndex(ix, iy - 1, iz)]) ||
                    (iz > 0 and !segmentation[IndicesToIndex(ix, iy, iz - 1)]) ||
                    (ix < grid_size[IB_X] - 1 and !segmentation[IndicesToIndex(ix + 1, iy, iz)]) ||
                    (iy < grid_size[IB_Y] - 1 and !segmentation[IndicesToIndex(ix, iy + 1, iz)]) ||
                    (iz < grid_size[IB_Z] - 1 and !segmentation[IndicesToIndex(ix, iy, iz + 1)])) {
                    b[IndicesToIndex(ix, iy, iz)] = 1;
                }
                else {
                    b[IndicesToIndex(ix, iy, iz)] = infinity;
                }
            }
        }
    }

    // go along the z dimenion first for every (x, y) coordinate
    for (long ix = 0; ix < grid_size[IB_X]; ++ix) {
        for (long iy = 0; iy < grid_size[IB_Y]; ++iy) {

            long k = 0;
            long *v = new long[grid_size[IB_Z] + 1];
            double *z = new double[grid_size[IB_Z] + 1];

            v[0] = 0;
            z[0] = -1 * infinity;
            z[1] = infinity;

            for (long q = 1; q < grid_size[IB_Z]; ++q) {
                // label for jump statement
                zlabel:
                double s = ((b[IndicesToIndex(ix, iy, q)] + q * q) - (b[IndicesToIndex(ix, iy, v[k])] + v[k] * v[k])) / (float)(2 * q - 2 * v[k]);
                
                if (s <= z[k]) {
                    --k;
                    goto zlabel;
                }
                else {
                    ++k;
                    v[k] = q;
                    z[k] = s;
                    z[k + 1] = infinity;
                }
            }

            k = 0;
            for (long q = 0; q < grid_size[IB_Z]; ++q) {
                while (z[k + 1] < q)
                    ++k;

                DBF[IndicesToIndex(ix, iy, q)] = (q - v[k]) * (q - v[k]) + b[IndicesToIndex(ix, iy, v[k])];
            }

            // free memory 
            delete[] v;
            delete[] z;
        }
    }

    // update the boundary values with this distance
    for (long iz = 0; iz < grid_size[IB_Z]; ++iz) {
        for (long iy = 0; iy < grid_size[IB_Y]; ++iy) {
            for (long ix = 0; ix < grid_size[IB_X]; ++ix) {
                b[IndicesToIndex(ix, iy, iz)] = DBF[IndicesToIndex(ix, iy, iz)];
            }
        }
    }

    // go along the y dimension second for every (z, x) coordinate
    for (long iz = 0; iz < grid_size[IB_Z]; ++iz) {
        for (long ix = 0; ix < grid_size[IB_X]; ++ix) {

            long k = 0;
            long *v = new long[grid_size[IB_Y] + 1];
            double *z = new double[grid_size[IB_Y] + 1];

            v[0] = 0;
            z[0] = -1 * infinity;
            z[1] = infinity;

            for (long q = 1; q < grid_size[IB_Y]; ++q) {
                // label for jump statement
                ylabel:
                double s = ((b[IndicesToIndex(ix, q, iz)] + q * q) - (b[IndicesToIndex(ix, v[k], iz)] +  v[k] * v[k])) / (float)(2 * q - 2 * v[k]);
                
                if (s <= z[k]) {
                    --k;
                    goto ylabel;
                }
                else {
                    ++k; 
                    v[k] = q;
                    z[k] = s;
                    z[k + 1] = infinity;
                }
            }

            k = 0;
            for (long q = 0; q < grid_size[IB_Y]; ++q) {
                while (z[k + 1] < q)
                    ++k;
            
                DBF[IndicesToIndex(ix, q, iz)] = (q - v[k]) * (q - v[k]) + b[IndicesToIndex(ix, v[k], iz)];
            }

            // free memory
            delete[] v;
            delete[] z;
        }
    }

    // update the boundary values with this distance
    for (long iz = 0; iz < grid_size[IB_Z]; ++iz) {
        for (long iy = 0; iy < grid_size[IB_Y]; ++iy) {
            for (long ix = 0; ix < grid_size[IB_X]; ++ix) {
                b[IndicesToIndex(ix, iy, iz)] = DBF[IndicesToIndex(ix, iy, iz)];
            }
        }
    }


    // go along the x dimension last for every (y, z) coordinate
    for (long iy = 0; iy < grid_size[IB_Y]; ++iy) {
        for (long iz = 0; iz < grid_size[IB_Z]; ++iz) {

            long k = 0;
            long *v = new long[grid_size[IB_X] + 1];
            double *z = new double[grid_size[IB_X] + 1];

            v[0] = 0;
            z[0] = -1 * infinity;
            z[1] = infinity;

            for (long q = 1; q < grid_size[IB_X]; ++q) {
                // label for jump statement
                xlabel:
                double s = ((b[IndicesToIndex(q, iy, iz)] + q * q) - (b[IndicesToIndex(v[k], iy, iz)] + v[k] * v[k])) / (float)(2 * q - 2 * v[k]);

                if (s <= z[k]) {
                    --k;
                    goto xlabel;
                }
                else {
                    ++k;
                    v[k] = q;
                    z[k] = s;
                    z[k + 1] = infinity;
                }
            }

            k = 0;
            for (long q = 0;  q < grid_size[IB_X]; ++q) {
                while (z[k + 1] < q)
                    ++k;

                DBF[IndicesToIndex(q, iy, iz)] = (q - v[k]) * (q - v[k]) + b[IndicesToIndex(v[k], iy, iz)];
            }

            // free memory
            delete[] v;
            delete[] z;
        }
    }

    for (long iv = 0; iv < nentries; ++iv) {
        DBF[iv] = sqrt(DBF[iv]);
    }

    // free memory
    delete[] b;
}



struct DijkstraData {
    long iv;
    DijkstraData *prev;
    double voxel_penalty;
    double distance;
    bool visited;
};



long ComputeDistanceFromVoxelField(long source_index)
{
    DijkstraData *voxel_data = new DijkstraData[nentries];
    if (!voxel_data) exit(-1);

    // initialize all data
    for (int iv = 0; iv < nentries; ++iv) {
        voxel_data[iv].iv = iv;
        voxel_data[iv].prev = NULL;
        voxel_data[iv].voxel_penalty = penalties[iv];
        voxel_data[iv].distance = infinity;
        voxel_data[iv].visited = false;
    }

    // initialize the priority queue
    DijkstraData tmp;
    MinBinaryHeap<DijkstraData *> voxel_heap(&tmp, (&tmp.distance), nentries);

    // insert the source into the heap
    voxel_data[source_index].distance = 0.0;
    voxel_data[source_index].visited = true;
    voxel_heap.Insert(source_index, &(voxel_data[source_index]));

    // visit all vertices
    long voxel_index = 0;
    while (!voxel_heap.IsEmpty()) {
        DijkstraData *current = voxel_heap.DeleteMin();
        voxel_index = current->iv;

        // visit all 26 neighbors of this index
        long ix, iy, iz;
        IndexToIndicies(voxel_index, ix, iy, iz);

        for (long iw = iz - 1; iw <= iz + 1; ++iw) {
            for (long iv = iy - 1; iv <= iy + 1; ++iv) {
                for (long iu = ix - 1; iu <= ix + 1; ++iu) {
                    // get the linear index for this voxel
                    long neighbor_index = IndicesToIndex(iu, iv, iw);

                    // skip if background
                    if (!segmentation[neighbor_index]) continue;
                    
                    // get the corresponding neighbor data
                    DijkstraData *neighbor_data = &(voxel_data[neighbor_index]);

                    // find the distance between these voxels
                    long deltaz = (iw - iz);
                    long deltay = (iv - iy);
                    long deltax = (iu - ix);

                    // get the distance between (ix, iy, iz) and (iu, iv, iw)
                    double distance = sqrt(deltax * deltax + deltay * deltay + deltaz * deltaz);

                    // get the distance to get to this voxel through the current voxel (requires a penalty for visiting this voxel)
                    double distance_through_current = current->distance + distance + neighbor_data->voxel_penalty;
                    double distance_without_current = neighbor_data->distance;

                    if (!neighbor_data->visited) {
                        neighbor_data->prev = current;
                        neighbor_data->distance = distance_through_current;
                        neighbor_data->visited = true;
                        voxel_heap.Insert(neighbor_index, neighbor_data);
                    }
                    else if (distance_through_current < distance_without_current) {
                        neighbor_data->prev = current;
                        neighbor_data->distance = distance_through_current;
                        voxel_heap.DecreaseKey(neighbor_index, neighbor_data);
                    }
                }
            }
        }
    }

    // first call to this function needs to return the root and does not compute the skeleton
    if (!PDRF) {
        // free memory
        delete[] voxel_data;

        // return the farthest voxel (to get the root voxel)
        return voxel_index;
    }

    // save the PDRF (only called when given root voxel)
    for (long iv = 0; iv < nentries; ++iv) {
        if (!segmentation[iv]) continue;
        PDRF[iv] = voxel_data[iv].distance;
    }

    // continue until there are no more inside voxels
    while (inside_voxels) {
        double farthest_pdrf = -1;
        long starting_voxel = -1;

        // find the farthest PDRF that is still inside
        for (long iv = 0; iv < nentries; ++iv) {
            if (!inside[iv]) continue;
            if (PDRF[iv] > farthest_pdrf) {
                farthest_pdrf = PDRF[iv];
                starting_voxel = iv;
            }
        }

        for (long iv = 0; iv < nentries; ++iv) {
            if (!inside[iv]) continue;
            long ix, iy, iz;
            IndexToIndicies(iv, ix, iy, iz);

            // get the skeleton path from this location to the root
            DijkstraData *current = &(voxel_data[starting_voxel]);

            while (!skeleton[current->iv]) {
                long ii, ij, ik;
                IndexToIndicies(current->iv, ii, ij, ik);
                // what is the distance between this skeleton location and the inside location
                double deltax = (ii - ix);
                double deltay = (ij - iy);
                double deltaz = (ik - iz);

                double distance = sqrt(deltax * deltax + deltay * deltay + deltaz * deltaz);

                if (distance < scale * DBF[current->iv] + buffer) {
                    inside[iv] = 0;
                    inside_voxels--;
                    break;
                }

                // update skeleton pointer
                current = current->prev;
            }
        }

        DijkstraData *current = &(voxel_data[starting_voxel]);
        while (!skeleton[current->iv]) {
            skeleton[current->iv] = 1;
            current = current->prev;
        }
    }

    // free memory
    delete[] voxel_data;

    return -1;
}



void ComputePenalties(void)
{
    // get the maximum distance from the boundary
    double M = 0;
    for (long iv = 0; iv < nentries; ++iv) {
        if (DBF[iv] > M) M = DBF[iv]; 
    }

    // choose 5000 so that 3000 length voxel paths have correct floating point precision
    const double pdrf_scale = 5000;
    for (long iv = 0; iv < nentries; ++iv) {
        penalties[iv] = pdrf_scale * pow(1 - DBF[iv] / M, 16);
    }
}



static bool IsEndpoint(long iv)
{
    long ix, iy, iz;
    IndexToIndicies(iv, ix, iy, iz);

    short nnneighbors = 0;
    for (long iw = iz - 1; iw <= iz + 1; ++iw) {
        for (long iv = iy - 1; iv <= iy + 1; ++iv) {
            for (long iu = ix - 1; iu <= ix + 1; ++iu) {
                long linear_index = IndicesToIndex(iu, iv, iw);
                if (segmentation[linear_index]) nnneighbors++;
            }
        }
    }

    // return if there is one neighbor (other than iv) that is 1
    if (nnneighbors <= 2) return true;
    else return false;
}



void CppTeaserSkeletonization(const char *prefix, long skeleton_resolution[3], bool benchmark, double input_scale, long input_buffer) 
{
    // set global variables
    scale = input_scale;
    buffer = input_buffer;

    // read the topologically downsampled file
    char input_filename[4096];
    if (benchmark) sprintf(input_filename, "benchmarks/skeleton/%s-downsample-%03ldx%03ldx%03ld.bytes", prefix, skeleton_resolution[IB_X], skeleton_resolution[IB_Y], skeleton_resolution[IB_Z]);
    else sprintf(input_filename, "skeletons/%s/downsample-%03ldx%03ldx%03ld.bytes", prefix, skeleton_resolution[IB_X], skeleton_resolution[IB_Y], skeleton_resolution[IB_Z]);

    // open the input file
    FILE *rfp = fopen(input_filename, "rb");
    if (!rfp) { fprintf(stderr, "Failed to read %s\n", input_filename); exit(-1); }

    // read the size and number of segments
    if (fread(&(grid_size[IB_Z]), sizeof(long), 1, rfp) != 1) { fprintf(stderr, "Failed to read %s\n", input_filename); exit(-1); }
    if (fread(&(grid_size[IB_Y]), sizeof(long), 1, rfp) != 1) { fprintf(stderr, "Failed to read %s\n", input_filename); exit(-1); }
    if (fread(&(grid_size[IB_X]), sizeof(long), 1, rfp) != 1) { fprintf(stderr, "Failed to read %s\n", input_filename); exit(-1); }

    // open the output filename
    char output_filename[4096];
    if (benchmark) sprintf(output_filename, "benchmarks/skeleton/%s-teaser-%03ldx%03ldx%03ld-downsample-%02ld-%02ld-skeleton.pts", prefix, skeleton_resolution[IB_X], skeleton_resolution[IB_Y], skeleton_resolution[IB_Z], (long)(10 * scale), buffer);
    else sprintf(output_filename, "skeletons/%s/teaser-%03ldx%03ldx%03ld-downsample-%02ld-%02ld-skeleton.pts", prefix, skeleton_resolution[IB_X], skeleton_resolution[IB_Y], skeleton_resolution[IB_Z], (long)(10 * scale), buffer);

    FILE *wfp = fopen(output_filename, "wb");
    if (!wfp) { fprintf(stderr, "Failed to write to %s\n", output_filename); exit(-1); }

    // go through all labels
    long max_label;
    if (fread(&max_label, sizeof(long), 1, rfp) != 1) { fprintf(stderr, "Failed to read %s\n", input_filename); exit(-1); }
    
    // write header for the skeleton file
    if (fwrite(&(grid_size[IB_Z]), sizeof(long), 1, wfp) != 1) { fprintf(stderr, "Failed to write to %s\n", output_filename); exit(-1); }
    if (fwrite(&(grid_size[IB_Y]), sizeof(long), 1, wfp) != 1) { fprintf(stderr, "Failed to write to %s\n", output_filename); exit(-1); }
    if (fwrite(&(grid_size[IB_X]), sizeof(long), 1, wfp) != 1) { fprintf(stderr, "Failed to write to %s\n", output_filename); exit(-1); }
    if (fwrite(&max_label, sizeof(long), 1, wfp) != 1) { fprintf(stderr, "Failed to write to %s\n", output_filename); exit(-1); }

    // add padding around each segment
    grid_size[IB_Z] += 2;
    grid_size[IB_Y] += 2;
    grid_size[IB_X] += 2;

    // set global indexing parameters
    nentries = grid_size[IB_Z] * grid_size[IB_Y] * grid_size[IB_X];
    sheet_size = grid_size[IB_Y] * grid_size[IB_X];
    row_size = grid_size[IB_X];
    infinity = grid_size[IB_Z] * grid_size[IB_Z] + grid_size[IB_Y] * grid_size[IB_Y] + grid_size[IB_X] * grid_size[IB_X];

    double *running_times = new double[max_label];

    for (long label = 0; label < max_label; ++label) {
        clock_t t1, t2;
        t1 = clock();

        // find the number of elements in this segment
        long num;
        if (fread(&num, sizeof(long), 1, rfp) != 1) { fprintf(stderr, "Failed to read %s\n", input_filename); exit(-1); }
        if (!num) { fwrite(&num, sizeof(long), 1, wfp); continue; }

        // allocate memory for global variables
        segmentation = new unsigned char[nentries];
        skeleton = new unsigned char[nentries];
        penalties = new double[nentries];
        inside = new unsigned char[nentries];
        DBF = new double[nentries];
        for (long iv = 0; iv < nentries; ++iv) {
            segmentation[iv] = 0;
            skeleton[iv] = 0;
            penalties[iv] = 0;
            inside[iv] = 0;
            DBF[iv] = 0;
        }

        for (long iv = 0; iv < num; ++iv) {
            long element;
            if (fread(&element, sizeof(long), 1, rfp) != 1) { fprintf(stderr, "Failed to read %s\n", input_filename); exit(-1); }


            // convert the element to non-cropped iz, iy, ix
            long iz = element / ((grid_size[IB_X] - 2) * (grid_size[IB_Y] - 2));
            long iy = (element - iz * (grid_size[IB_X] - 2) * (grid_size[IB_Y] - 2)) / (grid_size[IB_X] - 2);
            long ix = element % (grid_size[IB_X] - 2);

            // convert to cropped linear index
            element = (iz + 1) * sheet_size + (iy + 1) * row_size + ix + 1;

            segmentation[element] = 1;
            inside[element] = 1;
            inside_voxels++;
        }

        ComputeDistanceFromBoundaryField();

        // set any voxel as the source
        long source_voxel = -1;
        for (long iv = 0; iv < nentries; ++iv)
            if (inside[iv]) { source_voxel = iv; break; }

        // find a root voxel which is guaranteed to be at an extrema point
        long root_voxel = ComputeDistanceFromVoxelField(source_voxel);
        skeleton[root_voxel] = 1;
        inside[root_voxel] = 0;
        inside_voxels--;

        ComputePenalties();
        PDRF = new double[nentries];
        ComputeDistanceFromVoxelField(root_voxel);

        num = 0;
        for (long iv = 0; iv < nentries; ++iv) {
            if (skeleton[iv]) num++;
        }

        // delete segmentation and point to skeleton to find which indices are endpoints
        delete[] segmentation;
        segmentation = skeleton;

        // write the number of elements
        if (fwrite(&num, sizeof(long), 1, wfp) != 1) { fprintf(stderr, "Failed to write to %s\n", output_filename); exit(-1); }

        for (long iv = 0; iv < nentries; ++iv) {
            if (!skeleton[iv]) continue;

            long ix, iy, iz;
            IndexToIndicies(iv, ix, iy, iz);
            --ix; --iy; --iz;

            long element = iz * (grid_size[IB_X] - 2) * (grid_size[IB_Y] - 2) + iy * (grid_size[IB_X] - 2) + ix;
            if (IsEndpoint(iv)) element = -1 * element;

            // endpoints get a negative value
            if (fwrite(&element, sizeof(long), 1, wfp) != 1) { fprintf(stderr, "Failed to write to %s\n", output_filename); exit(-1); }
        }

        // free memory
        delete[] skeleton;
        delete[] penalties;
        delete[] PDRF;
        delete[] inside;
        delete[] DBF;

        // reset global variables
        segmentation = NULL;
        skeleton = NULL;
        penalties = NULL;
        PDRF = NULL;
        inside = NULL;
        DBF = NULL;

        t2 = clock();

        running_times[label] = (double)(t2 - t1) / CLOCKS_PER_SEC;
    }

    // close the files
    fclose(rfp);
    fclose(wfp);

    // save running time information
    if (benchmark) {
        char running_times_filename[4096];
        sprintf(running_times_filename, "benchmarks/skeleton/running-times/skeleton-times/%s-teaser-%03ldx%03ldx%03ld-%02ld-%02ld.bytes", prefix, skeleton_resolution[IB_X], skeleton_resolution[IB_Y], skeleton_resolution[IB_Z], (long)(10 * scale), buffer);

        FILE *running_times_fp = fopen(running_times_filename, "wb");
        if (!running_times_fp) exit(-1);
       
        if (fwrite(&max_label, sizeof(long), 1, running_times_fp) != 1) { fprintf(stderr, "Failed to write to %s\n", running_times_filename); }
        if (fwrite(running_times, sizeof(double), max_label, running_times_fp) != (unsigned long) max_label) { fprintf(stderr, "Failed to write to %s\n", running_times_filename); }

        fclose(running_times_fp);
    }


    delete[] running_times;
}
