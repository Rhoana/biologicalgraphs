import os
import struct

import scipy.spatial, scipy.optimize
import numpy as np

from ibex.utilities import dataIO
from ibex.utilities.constants import *



def ReadSkeletonEndpoints(filename):
    endpoints = []

    with open(filename, 'rb') as fd:
        zres, yres, xres, max_label, = struct.unpack('qqqq', fd.read(32))

        # go through every label
        for label in range(max_label):
            nelements, = struct.unpack('q', fd.read(8))

            endpoints.append([])

            # go through all elements
            for _ in range(nelements):
                index, = struct.unpack('q', fd.read(8))
                if index > 0: continue
                index = -1 * index

                # convert to cartesian coordinates
                iz = index / (yres * xres)
                iy = (index - iz * yres * xres) / xres
                ix = index % xres

                endpoints[label].append((ix, iy, iz))

    return endpoints



def ReadGroundTruth(prefix, max_label):
    examples_filename = 'benchmarks/skeleton/{}-skeleton-benchmark-examples.bin'.format(prefix)

    gt_examples = [[] for _ in range(max_label)]

    with open(examples_filename, "rb") as fd: 
        cutoff, = struct.unpack('q', fd.read(8))
        for iv in range(cutoff):
            label, = struct.unpack('q', fd.read(8))

            # read all the examples
            example_filename = 'benchmarks/skeleton/{}/skeleton-endpoints-{:05d}.pts'.format(prefix, label)
            if not os.path.exists(example_filename): continue

            with open(example_filename, 'rb') as efd:
                npts, = struct.unpack('q', efd.read(8))
                for _ in range(npts):
                    zpt, ypt, xpt, = struct.unpack('qqq', efd.read(24))

                    gt_examples[label].append((xpt, ypt, zpt))

    # read the list
    return gt_examples



def FindEndpointMatches(prefix, algorithm, params, resolution, ground_truth):
    # read the endpoints for this set of parameters
    skeleton_filename = 'benchmarks/skeleton/{}-{}-{:03d}x{:03d}x{:03d}-upsample-{}-skeleton.pts'.format(prefix, algorithm, resolution[IB_X], resolution[IB_Y], resolution[IB_Z], params)
    if not os.path.exists(skeleton_filename): return 0, 0, 0

    # read the endpoints
    proposed = ReadSkeletonEndpoints(skeleton_filename)
    assert (len(ground_truth) == len(proposed))

    # don't allow points to be connected over this distance
    max_distance = 800
    
    # go through every label
    max_label = len(ground_truth)

    output_filename = 'benchmarks/skeleton/matchings/{}-{}-{:03d}x{:03d}x{:03d}-{}-matching-pairs.pts'.format(prefix, algorithm, resolution[IB_X], resolution[IB_Y], resolution[IB_Z], params)

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    with open(output_filename, 'wb') as fd:
        # need resolution for max distance
        resolution = dataIO.Resolution(prefix)

        fd.write(struct.pack('q', max_label))
        for label in range(max_label):
            # no ground truth for this label
            if not len(ground_truth[label]): 
                fd.write(struct.pack('q', 0))
                continue

            ngt_pts = len(ground_truth[label])
            npr_pts = len(proposed[label])

            gt_pts = np.zeros((ngt_pts, 3), dtype=np.int64)
            pr_pts = np.zeros((npr_pts, 3), dtype=np.int64)

            # can not use IB_NDIMS because coordinates are (x, y, z) here
            for pt in range(ngt_pts):
                gt_pts[pt,0] = resolution[IB_X] * ground_truth[label][pt][0]
                gt_pts[pt,1] = resolution[IB_Y] * ground_truth[label][pt][1]
                gt_pts[pt,2] = resolution[IB_Z] * ground_truth[label][pt][2]

            for pt in range(npr_pts):
                pr_pts[pt,0] = resolution[IB_X] * proposed[label][pt][0]
                pr_pts[pt,1] = resolution[IB_Y] * proposed[label][pt][1]
                pr_pts[pt,2] = resolution[IB_Z] * proposed[label][pt][2]

            cost_matrix = scipy.spatial.distance.cdist(gt_pts, pr_pts)
            matching = scipy.optimize.linear_sum_assignment(cost_matrix)

            valid_matches = set()
            for match in zip(matching[0], matching[1]):
                # valid pairs must be within max_distance (in nanometers)
                if cost_matrix[match[0], match[1]] > max_distance: continue

                valid_matches.add(match)

            true_positives += len(valid_matches)
            false_positives += npr_pts - len(valid_matches)
            false_negatives += ngt_pts - len(valid_matches)

            # write the ground truth and the corresponding segment endpoints
            fd.write(struct.pack('q', len(valid_matches)))
            for match in valid_matches:
                fd.write(struct.pack('qq', match[0], match[1]))

    precision = true_positives / float(true_positives + false_positives)
    recall = true_positives / float(true_positives + false_negatives)

    fscore = 2 * (precision * recall) / float(precision + recall)

    return fscore, precision, recall    



def EvaluateEndpoints(prefix):
    gold = dataIO.ReadGoldData(prefix)
    max_label = np.amax(gold) + 1

    resolutions = [(iv, iv, iv) for iv in range(30, 210, 10)]           # all downsampled resolutions

    # get the human labeled ground truth
    gt_endpoints = ReadGroundTruth(prefix, max_label)

    best_fscore_precision = 0.0
    best_fscore_recall = 0.0
    best_fscore = 0.0
    algorithm = ''

    min_precision, min_recall =  (0.80, 0.90)

    # go through all possible configurations
    for resolution in resolutions:
        # go through parameters for medial axis strategy
        for astar_expansion in [0, 11, 13, 15, 17, 19, 21, 23, 25]:
            fscore, precision, recall = FindEndpointMatches(prefix, 'thinning', '{:02d}'.format(astar_expansion), resolution, gt_endpoints)

            if (precision > min_precision and recall > min_recall):
                print 'Thinning {:03d}x{:03d}x{:03d} {:02d}'.format(resolution[IB_X], resolution[IB_Y], resolution[IB_Z], astar_expansion)
                print '  F1-Score: {}'.format(fscore)
                print '  Precision: {}'.format(precision)
                print '  Recall: {}'.format(recall)

            if (fscore > best_fscore): 
                best_fscore = fscore
                best_fscore_precision = precision
                best_fscore_recall = recall
                algorithm = 'thinning-{:03d}x{:03d}x{:03d}-{:02d}'.format(resolution[IB_X], resolution[IB_Y], resolution[IB_Z], astar_expansion)
            

            fscore, precision, recall = FindEndpointMatches(prefix, 'medial-axis', '{:02d}'.format(astar_expansion), resolution, gt_endpoints)

            if (precision > min_precision and recall > min_recall):
                print 'Medial Axis {:03d}x{:03d}x{:03d} {:02d}'.format(resolution[IB_X], resolution[IB_Y], resolution[IB_Z], astar_expansion)
                print '  F1-Score: {}'.format(fscore)
                print '  Precision: {}'.format(precision)
                print '  Recall: {}'.format(recall)

            if (fscore > best_fscore): 
                best_fscore = fscore
                best_fscore_precision = precision
                best_fscore_recall = recall
                algorithm = 'medial-axis-{:03d}x{:03d}x{:03d}-{:02d}'.format(resolution[IB_X], resolution[IB_Y], resolution[IB_Z], astar_expansion)

        for tscale in [7, 9, 11, 13, 15, 17]:
            for tbuffer in [1, 2, 3, 4, 5]:
                fscore, precision, recall = FindEndpointMatches(prefix, 'teaser', '{:02d}-{:02d}-00'.format(tscale, tbuffer), resolution, gt_endpoints)
                
                if (precision > min_precision and recall > min_recall):
                    print 'TEASER {:03d}x{:03d}x{:03d} {:02d} {:02d}'.format(resolution[IB_X], resolution[IB_Y], resolution[IB_Z], tscale, tbuffer)
                    print '  F1-Score: {}'.format(fscore)
                    print '  Precision: {}'.format(precision)
                    print '  Recall: {}'.format(recall)


                if (fscore > best_fscore): 
                    best_fscore = fscore
                    best_fscore_precision = precision
                    best_fscore_recall = recall
                    algorithm = 'teaser-{:03d}x{:03d}x{:03d}-{:02d}-{:02d}-00'.format(resolution[IB_X], resolution[IB_Y], resolution[IB_Z], tscale, tbuffer)


    print 'Best method: {}'.format(algorithm)
    print 'F1-Score: {}'.format(best_fscore)
    print 'Precision: {}'.format(best_fscore_precision)
    print 'Recall: {}'.format(best_fscore_recall)


# find skeleton benchmark information
def GenerateExamples(prefix, cutoff=500):
    gold = dataIO.ReadGoldData(prefix)
    labels, counts = np.unique(gold, return_counts=True)

    filename = 'benchmarks/skeleton/{}-skeleton-benchmark-examples.bin'.format(prefix)
    with open(filename, 'wb') as fd:
        fd.write(struct.pack('q', cutoff))
        if labels[0] == 0: cutoff += 1
        for ie, (count, label) in enumerate(sorted(zip(counts, labels), reverse=True)):
            if not label: continue
            # don't include more than cutoff examples
            if ie == cutoff: break
            fd.write(struct.pack('q', label))
