import sys

from ibex.skeletonization import generate_skeletons

assert (len(sys.argv) == 3)

prefix = 'Kasthuri-one'
resolution = (int(sys.argv[1]), int(sys.argv[1]), int(sys.argv[1]))
astar_expansion = float(sys.argv[2])

generate_skeletons.MedialAxis(prefix, resolution, benchmark=True, astar_expansion=astar_expansion)
