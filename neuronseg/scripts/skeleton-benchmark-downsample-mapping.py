import sys


from biologicalgraphs.utilities import dataIO
from biologicalgraphs.transforms import seg2seg



assert (len(sys.argv) == 2)

prefix = 'Kasthuri-one'
resolution = (int(sys.argv[1]), int(sys.argv[1]), int(sys.argv[1]))

gold = dataIO.ReadGoldData(prefix)

seg2seg.DownsampleMapping(prefix, gold, resolution)
