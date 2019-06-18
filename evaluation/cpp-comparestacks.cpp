#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <unordered_map>
#include <unordered_set>




// constant variables

static const int IB_Z = 0;
static const int IB_Y = 1;
static const int IB_X = 2;




static long NChoose2(long N) 
{
    return N * (N - 1) / 2;
}



double *CppEvaluate(long *segmentation, long *gold, long grid_size[3], long *ground_truth_masks, long nmasks)
{
    // get convenient variables
    long nentries = grid_size[IB_Z] * grid_size[IB_Y] * grid_size[IB_X];

    // create a set of invalid ground truth locations
    std::unordered_set<long> masked_gold_labels = std::unordered_set<long>();
    for (long iv = 0; iv < nmasks; ++iv)
        masked_gold_labels.insert(ground_truth_masks[iv]);

    // update the number of nonzero if mask is on 
    long nnonzero = 0;
    for (long iv = 0; iv < nentries; ++iv) {
        if (masked_gold_labels.find(gold[iv]) == masked_gold_labels.end()) nnonzero++;
    }

    // get the maximum value for the segmentation and gold volumes
    long max_segment = 0;
    long max_gold = 0;
    for (long iv = 0; iv < nentries; ++iv) {
        if (segmentation[iv] > max_segment) max_segment = segmentation[iv];
        if (gold[iv] > max_gold) max_gold = gold[iv];
    }
    ++max_segment;
    ++max_gold;

    // create mappings from ij, i, and j to number of elements
    std::unordered_map<long, long> c = std::unordered_map<long, long>();
    std::unordered_map<long, long> s = std::unordered_map<long, long>();
    std::unordered_map<long, long> t = std::unordered_map<long, long>();
    for (long iv = 0; iv < nentries; ++iv) {
        if (masked_gold_labels.find(gold[iv]) != masked_gold_labels.end()) continue;

        c[segmentation[iv] * max_gold + gold[iv]]++;
        s[segmentation[iv]]++;
        t[gold[iv]]++;
    }

    long TP = 0;
    std::unordered_map<long, long>::iterator it;
    for (it = c.begin(); it != c.end(); ++it) {
        TP += NChoose2(it->second);
    }
    long TP_FP = 0;
    for (it = s.begin(); it != s.end(); ++it) {
        TP_FP += NChoose2(it->second);
    }
    long TP_FN = 0;
    for (it = t.begin(); it != t.end(); ++it) {
        TP_FN += NChoose2(it->second);
    }
    long FP = TP_FP - TP;
    long FN = TP_FN - TP;

    double VI_split = 0.0;
    double VI_merge = 0.0;

    for (it = c.begin(); it != c.end(); ++it) {
        long index = it->first;

        // get the segmentation and gold variables
        long is = index / max_gold;
        long ig = index % max_gold;

        double spi = s[is] / (double)nnonzero;
        double tpj = t[ig] / (double)nnonzero;
        double pij = it->second / (double)nnonzero;

        VI_split -= pij * log2(pij / tpj);
        VI_merge -= pij * log2(pij / spi);
    }

    // populate the results array and return
    double *results = new double[4];
    results[0] = FP / (double) (NChoose2(nnonzero));
    results[1] = FN / (double) (NChoose2(nnonzero));
    results[2] = VI_merge;
    results[3] = VI_split;

    return results;
}