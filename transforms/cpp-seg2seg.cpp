#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <queue>
#include <unordered_set>
#include <map>
#include <ctime>


#define IB_Z 0
#define IB_Y 1
#define IB_X 2


static long nentries;
static long row_size;
static long sheet_size;



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



void CppMapLabels(long *segmentation, long *mapping, unsigned long input_nentries)
{
    for (unsigned long iv = 0; iv < input_nentries; ++iv) {
        segmentation[iv] = mapping[segmentation[iv]];
    }
}



void CppRemoveSmallConnectedComponents(long *segmentation, int threshold, unsigned long input_nentries)
{
    if (threshold == 0) return;

    // find the maximum label
    long max_segment_label = 0;
    for (unsigned long iv = 0; iv < input_nentries; ++iv) {
        if (segmentation[iv] > max_segment_label) max_segment_label = segmentation[iv];
    }
    max_segment_label++;

    // create a counter array for the number of voxels
    int *nvoxels_per_segment = new int[max_segment_label];
    for (long iv = 0; iv < max_segment_label; ++iv) {
        nvoxels_per_segment[iv] = 0;
    }

    // count the number of voxels per segment
    for (unsigned long iv = 0; iv < input_nentries; ++iv) {
        nvoxels_per_segment[segmentation[iv]]++;
    }

    // create the array for the updated segmentation
    for (unsigned long iv = 0; iv < input_nentries; ++iv) {
        if (nvoxels_per_segment[segmentation[iv]] < threshold) segmentation[iv] = 0;
    }

    // free memory
    delete[] nvoxels_per_segment;
}




void CppForceConnectivity(long *segmentation, long grid_size[3])
{
    // create the new components array
    nentries = grid_size[IB_Z] * grid_size[IB_Y] * grid_size[IB_X];
    sheet_size = grid_size[IB_Y] * grid_size[IB_X];
    row_size = grid_size[IB_X];

    long *components = new long[nentries];
    for (long iv = 0; iv < nentries; ++iv)
        components[iv] = 0;

    // create the queue of labels
    std::queue<unsigned long> pixels = std::queue<unsigned long>();

    long current_index = 0;
    long current_label = 1;

    while (current_index < nentries) {
        // set this component and add to the queue
        components[current_index] = current_label;
        pixels.push(current_index);

        // iterate over all pixels in the queue
        while (pixels.size()) {
            // remove this pixel from the queue
            unsigned long pixel = pixels.front();
            pixels.pop();

            // add the six neighbors to the queue
            long iz, iy, ix;
            IndexToIndicies(pixel, ix, iy, iz);

            for (long iw = iz - 1; iw <= iz + 1; ++iw) {
                if (iw < 0 or iw >= grid_size[IB_Z]) continue;
                for (long iv = iy - 1; iv <= iy + 1; ++iv) {
                    if (iv < 0 or iv >= grid_size[IB_Y]) continue;
                    for (long iu = ix - 1; iu <= ix + 1; ++iu) {
                        if (iu < 0 or iu >= grid_size[IB_X]) continue;
                        long neighbor = IndicesToIndex(iu, iv, iw);

                        if (segmentation[pixel] == segmentation[neighbor] && !components[neighbor]) {
                            components[neighbor] = current_label;
                            pixels.push(neighbor);
                        }
                    }        
                }
            }
        }
        current_label++;

        // if the current index is already labeled, continue
        while (current_index < nentries && components[current_index]) current_index++;
    }

    // create a list of mappings
    long max_segment = 0;
    long max_component = 0;
    for (long iv = 0; iv < nentries; ++iv) {
        if (segmentation[iv] > max_segment) max_segment = segmentation[iv];
        if (components[iv] > max_component) max_component = components[iv];
    }
    max_segment++;
    max_component++;

    std::unordered_set<long> *seg2comp = new std::unordered_set<long>[max_segment];
    for (long iv = 0; iv < max_segment; ++iv)
        seg2comp[iv] = std::unordered_set<long>();

    // see if any segments have multiple components
    for (long iv = 0; iv < nentries; ++iv) {
        seg2comp[segmentation[iv]].insert(components[iv]);
    }

    long overflow = max_segment;
    long *comp2seg = new long[max_component];
    for (long iv = 1; iv < max_segment; ++iv) {
        if (seg2comp[iv].size() == 1) {
            // get the component for this segment
            long component = *(seg2comp[iv].begin());
            comp2seg[component] = iv;
        }
        else {
            // iterate over the set
            for (std::unordered_set<long>::iterator it = seg2comp[iv].begin(); it != seg2comp[iv].end(); ++it) {
                long component = *it;

                // one of the components keeps the label
                if (it == seg2comp[iv].begin()) comp2seg[component] = iv;
                // set the component to start at max_segment and increment
                else {
                    comp2seg[component] = overflow;
                    ++overflow;
                }
            }
        }
    }

    // update the segmentation
    for (long iv = 0; iv < nentries; ++iv) {
        if (!segmentation[iv]) segmentation[iv] = 0;
        else segmentation[iv] = comp2seg[components[iv]];
    }

    // free memory
    delete[] seg2comp;
    delete[] comp2seg;
    delete[] components;
}



void CppDownsampleMapping(const char *prefix, long *segmentation, float input_resolution[3], long output_resolution[3], long input_grid_size[3])
{
    // get the number of entries 
    long input_nentries = input_grid_size[IB_Z] * input_grid_size[IB_Y] * input_grid_size[IB_X];

    // get downsample ratios
    float zdown = ((float) output_resolution[IB_Z]) / input_resolution[IB_Z];
    float ydown = ((float) output_resolution[IB_Y]) / input_resolution[IB_Y];
    float xdown = ((float) output_resolution[IB_X]) / input_resolution[IB_X];

    // get the output resolution size
    long output_grid_size[3];
    output_grid_size[IB_Z] = (long) ceil(input_grid_size[IB_Z] / zdown);
    output_grid_size[IB_Y] = (long) ceil(input_grid_size[IB_Y] / ydown);
    output_grid_size[IB_X] = (long) ceil(input_grid_size[IB_X] / xdown);
    long output_sheet_size = output_grid_size[IB_Y] * output_grid_size[IB_X];
    long output_row_size = output_grid_size[IB_X];

    long max_segment = 0;
    for (long iv = 0; iv < input_nentries; ++iv)
        if (segmentation[iv] > max_segment) max_segment = segmentation[iv];
    max_segment++;

    // create a set for each segment of downsampled locations
    std::unordered_set<long> *downsample_sets = new std::unordered_set<long>[max_segment];
    for (long iv = 0; iv < max_segment; ++iv)
        downsample_sets[iv] = std::unordered_set<long>();

    long index = 0;
    for (long iz = 0; iz < input_grid_size[IB_Z]; ++iz) {
        for (long iy = 0; iy < input_grid_size[IB_Y]; ++iy) {
            for (long ix = 0; ix < input_grid_size[IB_X]; ++ix, ++index) {
                long segment = segmentation[index];
                if (!segment) continue;

                long iw = (long) (iz / zdown);
                long iv = (long) (iy / ydown);
                long iu = (long) (ix / xdown);

                long downsample_index = iw * output_sheet_size + iv * output_row_size + iu;
                downsample_sets[segment].insert(downsample_index);
            }
        }
    }

    // write the downsampling information
    char downsample_filename[4096];
    sprintf(downsample_filename, "skeletons/%s/downsample-%03ldx%03ldx%03ld.bytes", prefix, output_resolution[IB_X], output_resolution[IB_Y], output_resolution[IB_Z]);

    // open the output file
    FILE *dfp = fopen(downsample_filename, "wb");
    if (!dfp) { fprintf(stderr, "Failed to write to %s\n", downsample_filename); exit(-1); }

    // write the upsampling information
    char upsample_filename[4096];
    sprintf(upsample_filename, "skeletons/%s/upsample-%03ldx%03ldx%03ld.bytes", prefix, output_resolution[IB_X], output_resolution[IB_Y], output_resolution[IB_Z]);

    // open the output file
    FILE *ufp = fopen(upsample_filename, "wb");
    if (!ufp) { fprintf(stderr, "Failed to write to %s\n", upsample_filename); exit(-1); }

    // write the number of segments
    fwrite(&output_grid_size[IB_Z], sizeof(long), 1, dfp);
    fwrite(&output_grid_size[IB_Y], sizeof(long), 1, dfp);
    fwrite(&output_grid_size[IB_X], sizeof(long), 1, dfp);
    fwrite(&max_segment, sizeof(long), 1, dfp);

    // write the output file size of the upsample version
    fwrite(&(input_grid_size[IB_Z]), sizeof(long), 1, ufp);
    fwrite(&(input_grid_size[IB_Y]), sizeof(long), 1, ufp);
    fwrite(&(input_grid_size[IB_X]), sizeof(long), 1, ufp);
    fwrite(&max_segment, sizeof(long), 1, ufp);

    // output values for downsampling
    for (long label = 0; label < max_segment; ++label) {
        // write the size for this set
        long nelements = downsample_sets[label].size();
        fwrite(&nelements, sizeof(long), 1, dfp);
        fwrite(&nelements, sizeof(long), 1, ufp);
        for (std::unordered_set<long>::iterator it = downsample_sets[label].begin(); it != downsample_sets[label].end(); ++it) {
            long element = *it;
            fwrite(&element, sizeof(long), 1, dfp);

            long iz = element / (output_grid_size[IB_Y] * output_grid_size[IB_X]);
            long iy = (element - iz * output_grid_size[IB_Y] * output_grid_size[IB_X]) / output_grid_size[IB_X];
            long ix = element % output_grid_size[IB_X];

            long zmin = (long) (zdown * iz);
            long ymin = (long) (ydown * iy);
            long xmin = (long) (xdown * ix);

            long zmax = std::min((long) ceil(zdown * (iz + 1) + 1), input_grid_size[IB_Z]);
            long ymax = std::min((long) ceil(ydown * (iy + 1) + 1), input_grid_size[IB_Y]);
            long xmax = std::min((long) ceil(xdown * (ix + 1) + 1), input_grid_size[IB_X]);

            double closest_to_center = input_grid_size[IB_Z] * input_grid_size[IB_Y] * input_grid_size[IB_X];
            long upsample_index = -1;

            long zcenter = (zmax + zmin) / 2;
            long ycenter = (ymax + ymin) / 2;
            long xcenter = (xmax + xmin) / 2;

            for (long iw = zmin; iw < zmax; ++iw) {
                for (long iv = ymin; iv < ymax; ++iv) {
                    for (long iu = xmin; iu < xmax; ++iu) {
                        long linear_index = iw * input_grid_size[IB_Y] * input_grid_size[IB_X] + iv * input_grid_size[IB_X] + iu;


                        // find the closest point to the center
                        if (segmentation[linear_index] != label) continue;

                        double distance = abs(iw - zcenter) + abs(iv - ycenter) + abs(iu - xcenter);
                        if (distance < closest_to_center) {
                            closest_to_center = distance;
                            upsample_index = linear_index;
                        }
                    }
                }
            }

            fwrite(&upsample_index, sizeof(long), 1, ufp);
        }
    }

    // close the file
    fclose(dfp);
    fclose(ufp);

    // free memory
    delete[] downsample_sets;
}