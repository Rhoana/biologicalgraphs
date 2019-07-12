import sys


from biologicalgraphs.utilities import dataIO
from biologicalgraphs.skeletonization import generate_skeletons

assert (len(sys.argv) == 3)

prefix = 'Kasthuri-one'
resolution = (int(sys.argv[1]), int(sys.argv[1]), int(sys.argv[1]))
astar_expansion = int(sys.argv[2])


gold = dataIO.ReadGoldData(prefix)

generate_skeletons.TopologicalThinning(prefix, gold, resolution, astar_expansion)
