void CppMapLabels(long *segmentation, long *mapping, unsigned long nentries);
void CppRemoveSmallConnectedComponents(long *segmentation, int threshold, unsigned long nentries);
void CppForceConnectivity(long *segmentation, long grid_size[3]);
void CppDownsampleMapping(const char *prefix, long *segmentation, float input_resolution[3], long output_resolution[3], long input_grid_size[3]);
