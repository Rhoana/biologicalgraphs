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
    sprintf(downsample_filename, "benchmarks/skeleton/%s-downsample-%03ldx%03ldx%03ld.bytes", prefix, output_resolution[IB_X], output_resolution[IB_Y], output_resolution[IB_Z]);

    // open the output file
    FILE *dfp = fopen(downsample_filename, "wb");
    if (!dfp) { fprintf(stderr, "Failed to write to %s\n", downsample_filename); exit(-1); }

    // write the upsampling information
    char upsample_filename[4096];
    sprintf(upsample_filename, "benchmarks/skeleton/%s-upsample-%03ldx%03ldx%03ld.bytes", prefix, output_resolution[IB_X], output_resolution[IB_Y], output_resolution[IB_Z]);

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