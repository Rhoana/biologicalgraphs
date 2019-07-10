void CppFindMiddleBoundaries(long *segmentation, long grid_size[3]);
void CppGetMiddleBoundaryLocation(long label_one, long label_two, float &zpoint, float &ypoint, float &xpoint);
void CppFindMeanAffinities(long *segmentation, float *affinities, long grid_size[3]);
float CppGetMeanAffinity(long label_one, long label_two);
long *CppRemoveSingletons(long *segmentation, long grid_size[3], float threshold);