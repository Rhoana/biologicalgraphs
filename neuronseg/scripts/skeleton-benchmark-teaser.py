import sys



from biologicalgraphs.utilities import dataIO
from biologicalgraphs.skeletonization import generate_skeletons




assert (len(sys.argv) == 4)

prefix = 'Kasthuri-one'
resolution = (int(sys.argv[1]), int(sys.argv[1]), int(sys.argv[1]))
tscale = float(sys.argv[2])
tbuffer = long(sys.argv[3])


gold = dataIO.ReadGoldData(prefix)


generate_skeletons.TEASER(prefix, gold, resolution, teaser_scale=tscale, teaser_buffer=tbuffer)
