/* c++ file to upsample the skeletons to full resolution */

#include <math.h>
#include <unordered_set>
#include <map>
#include <queue>
#include <set>
#include <string.h>
#include "cpp-generate_skeletons.h"



// global variables for upsampling operation

static std::map<long, long> *down_to_up;
static long *segmentation;
static unsigned char *skeleton;



// convenient variables for moving between high and low resolutions

static float zdown;
static float ydown;
static float xdown;

static long up_grid_size[3];
static long up_nentries;
static long up_sheet_size;
static long up_row_size;

static long down_grid_size[3];
static long down_nentries;
static long down_sheet_size;
static long down_row_size;



// conver the index to indices
static void IndexToIndices(long iv, long &ix, long &iy, long &iz)
{
    iz = iv / down_sheet_size;
    iy = (iv - iz * down_sheet_size) / down_row_size;
    ix = iv % down_row_size;
}



static int MapDown2Up(const char *prefix, long skeleton_resolution[3])
{
    // get the downsample filename
    char downsample_filename[4096];
    sprintf(downsample_filename, "skeletons/%s/downsample-%03ldx%03ldx%03ld.bytes", prefix, skeleton_resolution[IB_X], skeleton_resolution[IB_Y], skeleton_resolution[IB_Z]);

    FILE *dfp = fopen(downsample_filename, "rb"); 
    if (!dfp) { fprintf(stderr, "Failed to read %s\n", downsample_filename); return 0; }

    // get the upsample filename
    char upsample_filename[4096];
    sprintf(upsample_filename, "skeletons/%s/upsample-%03ldx%03ldx%03ld.bytes", prefix, skeleton_resolution[IB_X], skeleton_resolution[IB_Y], skeleton_resolution[IB_Z]);

    FILE *ufp = fopen(upsample_filename, "rb");
    if (!ufp) { fprintf(stderr, "Failed to read %s\n", upsample_filename); return 0; }

    // read downsample header
    long down_max_segment;
    if (fread(&(down_grid_size[IB_Z]), sizeof(long), 1, dfp) != 1) { fprintf(stderr, "Failed to read %s\n", downsample_filename); return 0; }
    if (fread(&(down_grid_size[IB_Y]), sizeof(long), 1, dfp) != 1) { fprintf(stderr, "Failed to read %s\n", downsample_filename); return 0; }
    if (fread(&(down_grid_size[IB_X]), sizeof(long), 1, dfp) != 1) { fprintf(stderr, "Failed to read %s\n", downsample_filename); return 0; }
    if (fread(&down_max_segment, sizeof(long), 1, dfp) != 1) { fprintf(stderr, "Failed to read %s\n", downsample_filename); return 0; }

    // read upsample header
    long up_max_segment;
    if (fread(&(up_grid_size[IB_Z]), sizeof(long), 1, ufp) != 1) { fprintf(stderr, "Failed to read %s\n", upsample_filename); return 0; }
    if (fread(&(up_grid_size[IB_Y]), sizeof(long), 1, ufp) != 1) { fprintf(stderr, "Failed to read %s\n", upsample_filename); return 0; }
    if (fread(&(up_grid_size[IB_X]), sizeof(long), 1, ufp) != 1) { fprintf(stderr, "Failed to read %s\n", upsample_filename); return 0; }
    if (fread(&up_max_segment, sizeof(long), 1, ufp) != 1) { fprintf(stderr, "Failed to read %s\n", upsample_filename); return 0; }

    down_to_up = new std::map<long, long>[up_max_segment];
    for (long label = 0; label < up_max_segment; ++label) {
        down_to_up[label] = std::map<long, long>();

        long down_nelements, up_nelements;
        if (fread(&down_nelements, sizeof(long), 1, dfp) != 1) { fprintf(stderr, "Failed to read %s\n", downsample_filename); return 0; }
        if (fread(&up_nelements, sizeof(long), 1, ufp) != 1) { fprintf(stderr, "Failed to read %s\n", upsample_filename); return 0; }

        long *down_elements = new long[down_nelements];
        long *up_elements = new long[up_nelements];
        if (fread(down_elements, sizeof(long), down_nelements, dfp) != (unsigned long)down_nelements) { fprintf(stderr, "Failed to read %s\n", downsample_filename); return 0; }
        if (fread(up_elements, sizeof(long), up_nelements, ufp) != (unsigned long)up_nelements) { fprintf(stderr, "Failed to read %s\n", upsample_filename); return 0; }

        for (long ie = 0; ie < down_nelements; ++ie)
            down_to_up[label][down_elements[ie]] = up_elements[ie];
    }

    fclose(dfp);
    fclose(ufp);

    return 1;
}



static void FindEndpointVector(long index, double &vx, double &vy, double &vz)
{
    std::vector<long> path_from_endpoint = std::vector<long>();
    path_from_endpoint.push_back(index);

    while (path_from_endpoint.size() < 4) {
        short nneighbors = 0;
        long only_neighbor = -1;
        
        long ix, iy, iz;
        IndexToIndices(index, ix, iy, iz);

        for (long iw = iz - 1; iw <= iz + 1; ++iw) {
            if (iw < 0 or iw >= down_grid_size[IB_Z]) continue;
            for (long iv = iy - 1; iv <= iy + 1; ++iv) {
                if (iv < 0 or iv >= down_grid_size[IB_Y]) continue;
                for (long iu = ix - 1; iu <= ix + 1; ++iu) {
                    if (iu < 0 or iu >= down_grid_size[IB_X]) continue;

                    long neighbor_index = iw * down_grid_size[IB_Y] * down_grid_size[IB_X] + iv * down_grid_size[IB_X] + iu;
                    if (!skeleton[neighbor_index]) continue;

                    // skip if the neighbor is this index (i.e., it is not a neighbor)
                    if (neighbor_index == index) continue;

                    nneighbors += 1;
                    only_neighbor = neighbor_index;
                }
            }
        }

        // if there were no neighbors break since there are no more endpoints
        if (not nneighbors) break;
        // if there are two neighbors break since there is a split
        else if (nneighbors > 1) break;  
        else {
            // mask out this skeleton point so next iteration works
            skeleton[index] = 0;
            // reset the index to the neighbors value
            index = only_neighbor;
            // add this neighbor to the path
            path_from_endpoint.push_back(index);
        }
    }

    // reset the skeletons
    for (unsigned long iv = 0; iv < path_from_endpoint.size(); ++iv) {
        skeleton[path_from_endpoint[iv]] = 1;
    }

    // find the vector
    if (path_from_endpoint.size() == 1) {
        vx = 0.0;
        vy = 0.0;
        vz = 0.0;

        // cannot normalize
        return;
    }
    else {
        long ix, iy, iz, ii, ij, ik;
        IndexToIndices(path_from_endpoint[0], ix, iy, iz);
        IndexToIndices(path_from_endpoint[path_from_endpoint.size() - 1], ii, ij, ik);

        vx = ix - ii;
        vy = iy - ij;
        vz = iz - ik;
    }

    // we do not change the coordinate system for the vectors from anisotropic to isotropic
    // this allows us to easily compute edges because we can multiply by the resolutions to convert
    // anisotropic coordinates to isotropic ones

    double normalization = sqrt(vx * vx + vy * vy + vz * vz);
    vx = vx / normalization;
    vy = vy / normalization;
    vz = vz / normalization;
}



void CppFindEndpointVectors(const char *prefix, long skeleton_resolution[3], float output_resolution[3])
{
    // get the mapping from downsampled locations to upsampled ones
    if (!MapDown2Up(prefix, skeleton_resolution)) return;

    // get downsample ratios
    zdown = ((float) skeleton_resolution[IB_Z]) / output_resolution[IB_Z];
    ydown = ((float) skeleton_resolution[IB_Y]) / output_resolution[IB_Y];
    xdown = ((float) skeleton_resolution[IB_X]) / output_resolution[IB_X];

    // set global variables
    up_nentries = up_grid_size[IB_Z] * up_grid_size[IB_Y] * up_grid_size[IB_X];
    up_sheet_size = up_grid_size[IB_Y] * up_grid_size[IB_X];
    up_row_size = up_grid_size[IB_X];

    down_nentries = down_grid_size[IB_Z] * down_grid_size[IB_Y] * down_grid_size[IB_X];
    down_sheet_size = down_grid_size[IB_Y] * down_grid_size[IB_X];
    down_row_size = down_grid_size[IB_X];

    // I/O filenames
    char input_filename[4096];
    sprintf(input_filename, "skeletons/%s/thinning-%03ldx%03ldx%03ld-downsample-skeleton.pts", prefix, skeleton_resolution[IB_X], skeleton_resolution[IB_Y], skeleton_resolution[IB_Z]);

    char output_filename[4096];
    sprintf(output_filename, "skeletons/%s/thinning-%03ldx%03ldx%03ld-endpoint-vectors.vec", prefix, skeleton_resolution[IB_X], skeleton_resolution[IB_Y], skeleton_resolution[IB_Z]);

    // open files for read/write
    FILE *rfp = fopen(input_filename, "rb");
    if (!rfp) { fprintf(stderr, "Failed to read %s\n", input_filename); return; }

    FILE *wfp = fopen(output_filename, "wb");
    if (!wfp) { fprintf(stderr, "Failed to write %s\n", output_filename); return; }

    // read header
    long max_label;
    long input_grid_size[3];
    if (fread(&(input_grid_size[IB_Z]), sizeof(long), 1, rfp) != 1) { fprintf(stderr, "Failed to read %s\n", input_filename); return; }
    if (fread(&(input_grid_size[IB_Y]), sizeof(long), 1, rfp) != 1) { fprintf(stderr, "Failed to read %s\n", input_filename); return; }
    if (fread(&(input_grid_size[IB_X]), sizeof(long), 1, rfp) != 1) { fprintf(stderr, "Failed to read %s\n", input_filename); return; }
    if (fread(&max_label, sizeof(long), 1, rfp) != 1) { fprintf(stderr, "Failed to read %s\n", input_filename); return; }
    
    // write the header
    if (fwrite(&(up_grid_size[IB_Z]), sizeof(long), 1, wfp) != 1) { fprintf(stderr, "Failed to write %s\n", output_filename); return; }
    if (fwrite(&(up_grid_size[IB_Y]), sizeof(long), 1, wfp) != 1) { fprintf(stderr, "Failed to write %s\n", output_filename); return; }
    if (fwrite(&(up_grid_size[IB_X]), sizeof(long), 1, wfp) != 1) { fprintf(stderr, "Failed to write %s\n", output_filename); return; }
    if (fwrite(&max_label, sizeof(long), 1, wfp) != 1) { fprintf(stderr, "Failed to write %s\n", output_filename); return; }

    for (long label = 0; label < max_label; ++label) {
        long nelements;
        if (fread(&nelements, sizeof(long), 1, rfp) != 1) { fprintf(stderr, "Failed to read %s\n", input_filename); return; }

        skeleton = new unsigned char[down_nentries];
        for (long iv = 0; iv < down_nentries; ++iv) skeleton[iv] = 0;

        // find all of the downsampled elements
        long *down_elements = new long[nelements];
        if (fread(down_elements, sizeof(long), nelements, rfp) != (unsigned long)nelements) { fprintf(stderr, "Failed to read %s\n", input_filename); return; }
        
        long nendpoints = 0;
        for (long ie = 0; ie < nelements; ++ie) {
            if (down_elements[ie] < 0) {
                skeleton[-1 * down_elements[ie]] = 1;
                nendpoints++;
            }
            else skeleton[down_elements[ie]] = 1;
        }
        if (fwrite(&nendpoints, sizeof(long), 1, wfp) != 1) { fprintf(stderr, "Failed to write to %s\n", output_filename); return; }

        // go through all down elements to find endpoints
        for (long ie = 0; ie < nelements; ++ie) {
            if (down_elements[ie] >= 0) continue;

            double vx, vy, vz;
            FindEndpointVector(-1 * down_elements[ie], vx, vy, vz);

            // get the corresponding up element for this endpoint
            long up_element = down_to_up[label][-1 * down_elements[ie]];

            // save the up element with the vector
            if (fwrite(&up_element, sizeof(long), 1, wfp) != 1) { fprintf(stderr, "Failed to write to %s\n", output_filename); return; }
            if (fwrite(&vz, sizeof(double), 1, wfp) != 1) { fprintf(stderr, "Failed to write to %s\n", output_filename); return; }
            if (fwrite(&vy, sizeof(double), 1, wfp) != 1) { fprintf(stderr, "Failed to write to %s\n", output_filename); return; }
            if (fwrite(&vx, sizeof(double), 1, wfp) != 1) { fprintf(stderr, "Failed to write to %s\n", output_filename); return; }
        }


        delete[] skeleton;
    }    

    // close the file
    fclose(rfp);
    fclose(wfp);

    delete[] down_to_up;
}



// operation that takes skeletons and 
void CppApplyUpsampleOperation(const char *prefix, long *input_segmentation, long skeleton_resolution[3], float output_resolution[3])
{
    // get the mapping from downsampled locations to upsampled ones
    if (!MapDown2Up(prefix, skeleton_resolution)) return;

    // get a list of labels for each downsampled index
    segmentation = input_segmentation;

    // get downsample ratios
    zdown = ((float) skeleton_resolution[IB_Z]) / output_resolution[IB_Z];
    ydown = ((float) skeleton_resolution[IB_Y]) / output_resolution[IB_Y];
    xdown = ((float) skeleton_resolution[IB_X]) / output_resolution[IB_X];

    // set global variables
    up_nentries = up_grid_size[IB_Z] * up_grid_size[IB_Y] * up_grid_size[IB_X];
    up_sheet_size = up_grid_size[IB_Y] * up_grid_size[IB_X];
    up_row_size = up_grid_size[IB_X];

    down_nentries = down_grid_size[IB_Z] * down_grid_size[IB_Y] * down_grid_size[IB_X];
    down_sheet_size = down_grid_size[IB_Y] * down_grid_size[IB_X];
    down_row_size = down_grid_size[IB_X];

    // I/O filenames
    char input_filename[4096];
    sprintf(input_filename, "skeletons/%s/thinning-%03ldx%03ldx%03ld-downsample-skeleton.pts", prefix, skeleton_resolution[IB_X], skeleton_resolution[IB_Y], skeleton_resolution[IB_Z]);
    
    char output_filename[4096];
    sprintf(output_filename, "skeletons/%s/thinning-%03ldx%03ldx%03ld-upsample-skeleton.pts", prefix, skeleton_resolution[IB_X], skeleton_resolution[IB_Y], skeleton_resolution[IB_Z]);

    // open files for read/write
    FILE *rfp = fopen(input_filename, "rb");
    if (!rfp) { fprintf(stderr, "Failed to read %s\n", input_filename); return; }

    FILE *wfp = fopen(output_filename, "wb");
    if (!wfp) { fprintf(stderr, "Failed to write %s\n", output_filename); return; }

    // read header
    long max_label;
    long input_grid_size[3];
    if (fread(&(input_grid_size[IB_Z]), sizeof(long), 1, rfp) != 1) { fprintf(stderr, "Failed to read %s\n", input_filename); return; }
    if (fread(&(input_grid_size[IB_Y]), sizeof(long), 1, rfp) != 1) { fprintf(stderr, "Failed to read %s\n", input_filename); return; }
    if (fread(&(input_grid_size[IB_X]), sizeof(long), 1, rfp) != 1) { fprintf(stderr, "Failed to read %s\n", input_filename); return; }
    if (fread(&max_label, sizeof(long), 1, rfp) != 1) { fprintf(stderr, "Failed to read %s\n", input_filename); return; }
    
    // write the header
    if (fwrite(&(up_grid_size[IB_Z]), sizeof(long), 1, wfp) != 1) { fprintf(stderr, "Failed to write %s\n", output_filename); return; }
    if (fwrite(&(up_grid_size[IB_Y]), sizeof(long), 1, wfp) != 1) { fprintf(stderr, "Failed to write %s\n", output_filename); return; }
    if (fwrite(&(up_grid_size[IB_X]), sizeof(long), 1, wfp) != 1) { fprintf(stderr, "Failed to write %s\n", output_filename); return; }
    if (fwrite(&max_label, sizeof(long), 1, wfp) != 1) { fprintf(stderr, "Failed to write %s\n", output_filename); return; }

    // go through all skeletons
    for (long label = 0; label < max_label; ++label) {
        long nelements;
        if (fread(&nelements, sizeof(long), 1, rfp) != 1) { fprintf(stderr, "Failed to read %s\n", input_filename); return; }
        if (fwrite(&nelements, sizeof(long), 1, wfp) != 1) { fprintf(stderr, "Failed to write %s\n", output_filename); return; }

        long *down_elements = new long[nelements];
        if (fread(down_elements, sizeof(long), nelements, rfp) != (unsigned long)nelements) { fprintf(stderr, "Failed to read %s\n", input_filename); return; }

        long *up_elements = new long[nelements];
        for (long ie = 0; ie < nelements; ++ie) {
            long down_index = down_elements[ie];

            if (down_index < 0) {
                down_index = -1 * down_index;
                up_elements[ie] = -1 * down_to_up[label][down_index];
            }
            else {
                up_elements[ie] = down_to_up[label][down_index];
            }
        }
        
        if (fwrite(up_elements, sizeof(long), nelements, wfp) != (unsigned long)nelements) { fprintf(stderr, "Failed to write %s\n", output_filename); return; }
        
        // free memory
        delete[] down_elements;
        delete[] up_elements;
    }

    // free memory
    delete[] down_to_up;

    // close the files
    fclose(rfp);
    fclose(wfp);
}
