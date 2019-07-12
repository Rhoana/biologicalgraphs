import sys

from ibex.skeletonization import generate_skeletons

assert (len(sys.argv) == 4)

prefix = 'Kasthuri-one'
resolution = (int(sys.argv[1]), int(sys.argv[1]), int(sys.argv[1]))
tscale = float(sys.argv[2])
tbuffer = long(sys.argv[3])

generate_skeletons.TEASER(prefix, resolution, benchmark=True, teaser_scale=tscale, teaser_buffer=tbuffer)
