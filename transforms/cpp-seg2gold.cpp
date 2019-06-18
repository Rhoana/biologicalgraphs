#include <stdio.h>
#include <unordered_map>



long *CppMapping(long *segmentation, long *gold, long nentries, double match_threshold, double nonzero_threshold)
{
    // find the maximum segmentation value
    long max_segmentation_value = 0;
    for (long iv = 0; iv < nentries; ++iv) {
        if (segmentation[iv] > max_segmentation_value)
            max_segmentation_value = segmentation[iv];
    }
    max_segmentation_value++;

    // find the maximum gold value
    long max_gold_value = 0;
    for (long iv = 0; iv < nentries; ++iv) {
        if (gold[iv] > max_gold_value) 
            max_gold_value = gold[iv];
    }
    max_gold_value++;

    // find the number of voxels per segment
    unsigned long *nvoxels_per_segment = new unsigned long[max_segmentation_value];
    for (long iv = 0; iv < max_segmentation_value; ++iv)
        nvoxels_per_segment[iv] = 0;
    for (long iv = 0; iv < nentries; ++iv)
        nvoxels_per_segment[segmentation[iv]]++;

    std::unordered_map<long, std::unordered_map<long, long> > seg2gold_overlap = std::unordered_map<long, std::unordered_map<long, long> >();   
    for (long is = 0; is < max_segmentation_value; ++is) {
        if (nvoxels_per_segment[is]) seg2gold_overlap.insert(std::pair<long, std::unordered_map<long, long> >(is, std::unordered_map<long, long>()));
    }
    
    for (long iv = 0; iv < nentries; ++iv) {
        seg2gold_overlap[segmentation[iv]][gold[iv]]++;
    }
    
    // create the mapping
    long *segmentation_to_gold = new long[max_segmentation_value];
    for (long is = 0; is < max_segmentation_value; ++is) {
        if (!nvoxels_per_segment[is]) { segmentation_to_gold[is] = 0; continue; }
        long gold_id = 0;
        long gold_max_value = 0;

        // only gets label of 0 if the number of non zero voxels is below threshold
        for (std::unordered_map<long, long>::iterator iter = seg2gold_overlap[is].begin(); iter != seg2gold_overlap[is].end(); ++iter) {
            if (not iter->first) continue;
            if (iter->second > gold_max_value) {
                gold_max_value = iter->second;
                gold_id = iter->first;
            }
        }

        // number of non zero pixels must be greater than the nonzero threshold
        if ((double)(nvoxels_per_segment[is] - seg2gold_overlap[is][0]) / nvoxels_per_segment[is] < nonzero_threshold) segmentation_to_gold[is] = 0;
        // the number of matching gold values divided by the number of non zero pixels must be greater than the match threshold or it is a merge error
        else if (gold_max_value / (double)(nvoxels_per_segment[is] - seg2gold_overlap[is][0]) < match_threshold) segmentation_to_gold[is] = -1;
        else segmentation_to_gold[is] = gold_id;
    }

    // free memory
    delete[] nvoxels_per_segment;
    
    return segmentation_to_gold;
}
