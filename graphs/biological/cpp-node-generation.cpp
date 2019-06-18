#include <stdio.h>
#include <unordered_map>



#define IB_Z 0
#define IB_Y 1
#define IB_X 2


// useful global variables

static long nentries;
static long sheet_size;
static long row_size;



// random access variables

static long max_label = -1;
static std::unordered_map<long, float> zmean = std::unordered_map<long, float>();
static std::unordered_map<long, float> ymean = std::unordered_map<long, float>();
static std::unordered_map<long, float> xmean = std::unordered_map<long, float>();


static std::unordered_map<long, float> mean_affinities = std::unordered_map<long, float>();




static long IndicesToIndex(long ix, long iy, long iz)
{
    return iz * sheet_size + iy * row_size + ix;
}




void CppFindMiddleBoundaries(long *segmentation, long grid_size[3])
{
    // set global variables
    nentries = grid_size[IB_Z] * grid_size[IB_Y] * grid_size[IB_X];
    sheet_size = grid_size[IB_Y] * grid_size[IB_X];
    row_size = grid_size[IB_X];

    // clear the maps -- needed for when this function is called twice in node-generation-runone
    zmean.clear();
    ymean.clear();
    xmean.clear();
        
    // clear the max_label to get an accurate maximum when called twice
    max_label = -1;

    // create mapping for the counts
    std::unordered_map<long, long> counts = std::unordered_map<long, long>();

    // find the largest label
    for (long ie = 0; ie < nentries; ++ie)
        if (segmentation[ie] > max_label) max_label = segmentation[ie];
    max_label++;
    if (max_label > (1L << 32) - 1) { fprintf(stderr, "CppFindMiddleBoundaries cannot handle int64 labels yet...\n"); exit(-1); }

    // go through all locations
    for (long iz = 0; iz < grid_size[IB_Z]; ++iz) {
        for (long iy = 0; iy < grid_size[IB_Y]; ++iy) {
            for (long ix = 0; ix < grid_size[IB_X]; ++ix) {
                long label = segmentation[IndicesToIndex(ix, iy, iz)];

                if (iz < grid_size[IB_Z] - 1) {
                    long neighbor = segmentation[IndicesToIndex(ix, iy, iz + 1)];
                    if (neighbor != label) {
                        long label_one = std::min(label, neighbor);
                        long label_two = std::max(label, neighbor);
                        zmean[label_one * max_label + label_two] += (iz + 0.5);
                        ymean[label_one * max_label + label_two] += iy;
                        xmean[label_one * max_label + label_two] += ix;
                        counts[label_one * max_label + label_two]++;
                    }
                }
                if (iy < grid_size[IB_Y] - 1) {
                    long neighbor = segmentation[IndicesToIndex(ix, iy + 1, iz)];
                    if (neighbor != label) {
                        long label_one = std::min(label, neighbor);
                        long label_two = std::max(label, neighbor);
                        zmean[label_one * max_label + label_two] += iz;
                        ymean[label_one * max_label + label_two] += (iy + 0.5);
                        xmean[label_one * max_label + label_two] += ix;
                        counts[label_one * max_label + label_two]++;
                    }
                }
                if (ix < grid_size[IB_X] - 1) {
                    long neighbor = segmentation[IndicesToIndex(ix + 1, iy, iz)];
                    if (neighbor != label) {
                        long label_one = std::min(label, neighbor);
                        long label_two = std::max(label, neighbor);
                        zmean[label_one * max_label + label_two] += iz;
                        ymean[label_one * max_label + label_two] += iy;
                        xmean[label_one * max_label + label_two] += (ix + 0.5);
                        counts[label_one * max_label + label_two]++;
                    }
                }
            }
        }
    }

    for (std::unordered_map<long, long>::iterator it = counts.begin(); it != counts.end(); ++it) {
        long index = it->first;

        zmean[index] /= counts[index];
        ymean[index] /= counts[index];
        xmean[index] /= counts[index];
    }
}



void CppGetMiddleBoundaryLocation(long label_one, long label_two, float &zpoint, float &ypoint, float &xpoint)
{
    zpoint = zmean[label_one * max_label + label_two];
    ypoint = ymean[label_one * max_label + label_two];
    xpoint = xmean[label_one * max_label + label_two];
}



void CppFindMeanAffinities(long *segmentation, float *affinities, long grid_size[3])
{
    // set global variables
    nentries = grid_size[IB_Z] * grid_size[IB_Y] * grid_size[IB_X];
    sheet_size = grid_size[IB_Y] * grid_size[IB_X];
    row_size = grid_size[IB_X];

    // clear the maps -- needed for when this function is called twice in node-generation-runone
    mean_affinities.clear();
            
    // clear the max_label to get an accurate maximum when called twice
    max_label = -1;

    // create mapping for the counts
    std::unordered_map<long, long> counts = std::unordered_map<long, long>();

    // find the largest label
    for (long ie = 0; ie < nentries; ++ie)
        if (segmentation[ie] > max_label) max_label = segmentation[ie];
    max_label++;
    if (max_label > (1L << 32) - 1) { fprintf(stderr, "CppFindMeanAffinities cannot handle int64 labels yet...\n"); exit(-1); }

    // go through all locations
    for (long iz = 0; iz < grid_size[IB_Z]; ++iz) {
        for (long iy = 0; iy < grid_size[IB_Y]; ++iy) {
            for (long ix = 0; ix < grid_size[IB_X]; ++ix) {
                long label = segmentation[IndicesToIndex(ix, iy, iz)];

                if (iz < grid_size[IB_Z] - 1) {
                    long neighbor = segmentation[IndicesToIndex(ix, iy, iz + 1)];
                    if (neighbor != label) {
                        long label_one = std::min(label, neighbor);
                        long label_two = std::max(label, neighbor);

                        mean_affinities[label_one * max_label + label_two] += affinities[3 * IndicesToIndex(ix, iy, iz) + 2];
                        counts[label_one * max_label + label_two]++;
                    }
                }
                if (iy < grid_size[IB_Y] - 1) {
                    long neighbor = segmentation[IndicesToIndex(ix, iy + 1, iz)];
                    if (neighbor != label) {
                        long label_one = std::min(label, neighbor);
                        long label_two = std::max(label, neighbor);

                        mean_affinities[label_one * max_label + label_two] += affinities[3 * IndicesToIndex(ix, iy, iz) + 1];
                        counts[label_one * max_label + label_two]++;
                    }
                }
                if (ix < grid_size[IB_X] - 1) {
                    long neighbor = segmentation[IndicesToIndex(ix + 1, iy, iz)];
                    if (neighbor != label) {
                        long label_one = std::min(label, neighbor);
                        long label_two = std::max(label, neighbor);

                        mean_affinities[label_one * max_label + label_two] += affinities[3 * IndicesToIndex(ix, iy, iz)];
                        counts[label_one * max_label + label_two]++;
                    }
                }
            }
        }
    }

    for (std::unordered_map<long, long>::iterator it = counts.begin(); it != counts.end(); ++it) {
        long index = it->first;

        mean_affinities[index] /= counts[index];
    }
}



float CppGetMeanAffinity(long label_one, long label_two)
{
  return mean_affinities[label_one * max_label + label_two];
}
