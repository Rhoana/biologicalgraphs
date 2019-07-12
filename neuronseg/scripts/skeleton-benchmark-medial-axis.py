import sys



from biologicalgraphs.utilities import dataIO
from biologicalgraphs.skeletonization import generate_skeletons



assert (len(sys.argv) == 3)



prefix = 'Kasthuri-one'
resolution = (int(sys.argv[1]), int(sys.argv[1]), int(sys.argv[1]))
astar_expansion = float(sys.argv[2])


gold = dataIO.ReadGoldData(prefix)


generate_skeletons.MedialAxis(prefix, gold, resolution, astar_expansion=astar_expansion)
