from biologicalgraphs.utilities import dataIO
from biologicalgraphs.graphs.biological import node_generation, edge_generation
from biologicalgraphs.evaluation import comparestacks
from biologicalgraphs.cnns.biological import edges, nodes
from biologicalgraphs.transforms import seg2seg, seg2gold
from biologicalgraphs.skeletonization import generate_skeletons
from biologicalgraphs.algorithms import lifted_multicut



# the prefix name corresponds to the meta file in meta/{PREFIX}.meta
prefix = 'Kasthuri-test'

# read the ground truth for this data
gold = dataIO.ReadGoldData(prefix)

# read the input segmentation data
segmentation = dataIO.ReadSegmentationData(prefix)

# subset is either training, validation, or testing
subset = 'testing'

# remove the singleton slices
node_generation.RemoveSingletons(prefix, segmentation)

# need to update the prefix and segmentation
# removesingletons writes a new h5 file to disk
prefix = '{}-segmentation-wos'.format(prefix)
segmentation = dataIO.ReadSegmentationData(prefix)
# need to rerun seg2gold mapping since segmentation changed
seg2gold_mapping = seg2gold.Mapping(prefix, segmentation, gold)


# generate locations for segments that are too small
node_generation.GenerateNodes(prefix, segmentation, subset, seg2gold_mapping)

# run inference for node network
node_model_prefix = 'architectures/nodes-400nm-3x20x60x60-Kasthuri/nodes'
nodes.forward.Forward(prefix, node_model_prefix, segmentation, subset, seg2gold_mapping, evaluate=True)

# need to update the prefix and segmentation
# node generation writes a new h5 file to disk
prefix = '{}-reduced-{}'.format(prefix, node_model_prefix.split('/')[1])
segmentation = dataIO.ReadSegmentationData(prefix)
# need to rerun seg2gold mapping since segmentation changed
seg2gold_mapping = seg2gold.Mapping(prefix, segmentation, gold)


# generate the skeleton by getting high->low resolution mappings
## and running topological thinnings
seg2seg.DownsampleMapping(prefix, segmentation)
generate_skeletons.TopologicalThinning(prefix, segmentation)
generate_skeletons.FindEndpointVectors(prefix)

# run edge generation function
edge_generation.GenerateEdges(prefix, segmentation, subset, seg2gold_mapping)

# run inference for edge network
edge_model_prefix = 'architectures/edges-600nm-3x18x52x52-Kasthuri/edges'
edges.forward.Forward(prefix, edge_model_prefix, subset)

# run lifted multicut
lifted_multicut.LiftedMulticut(prefix, segmentation, edge_model_prefix, seg2gold_mapping)
