#ifndef __CPP_GENERATE_SKELETONS__
#define __CPP_GENERATE_SKELETONS__

// function calls across cpp files
void CppTopologicalThinning(const char *prefix, long skeleton_resolution[3], const char *lookup_table_directory);
void CppTeaserSkeletonization(const char *prefix, long skeleton_resolution[3], double input_scale, long input_buffer);
void CppApplyUpsampleOperation(const char *prefix, const char *params, long *input_segmentation, long skeleton_resolution[3], float output_resolution[3], const char *skeleton_algorithm, double astar_expansion);


// universal variables and functions

static const int IB_Z = 0;
static const int IB_Y = 1;
static const int IB_X = 2;

#endif