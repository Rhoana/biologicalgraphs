import sys

from ibex.transforms import seg2seg

assert (len(sys.argv) == 2)

prefix = 'Kasthuri-one'
resolution = (int(sys.argv[1]), int(sys.argv[1]), int(sys.argv[1]))

output_filename = 'benchmarks/skeleton/running-times/{}-downsample-mapping-{:03d}x{:03d}x{:03d}.txt'.format(prefix, resolution[0], resolution[1], resolution[2])

sys.stdout = open(output_filename, 'w')

seg2seg.DownsampleMapping(prefix, resolution, benchmark=True)
