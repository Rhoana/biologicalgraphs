import sys

from biologicalgraphs.utilities.constants import *


class MetaData:
    def __init__(self, prefix):
        # initialize the prefix variable
        self.prefix = prefix

        # backwards compatability variable defaults
        self.affinity_filename = None
        self.gold_filename = None
        self.segment_filename = None
        self.crop_xmin = None
        self.crop_xmax = None
        self.crop_ymin = None
        self.crop_ymax = None
        self.crop_zmin = None
        self.crop_zmax = None

        # open the meta data txt file
        filename = 'meta/{}.meta'.format(prefix)
        with open(filename, 'r') as fd:
            lines = fd.readlines()

            for ix in range(0, len(lines), 2):
                # get the comment and the corresponding value
                comment = lines[ix].strip()
                value = lines[ix + 1].strip()

                if comment == '# resolution in nm':
                    # separate into individual dimensions
                    samples = value.split('x')
                    # need to use 2, 1, and 0 here since the outward facing convention is x,y,z, not z, y, x
                    self.resolution = (float(samples[2]), float(samples[1]), float(samples[0]))
                elif comment == '# affinity filename':
                    self.affinity_filename = value
                elif comment == '# gold filename':
                    self.gold_filename = value
                elif comment == '# segmentation filename':
                    self.segment_filename = value
                elif comment == '# grid size':
                    # read the grid size in x, y, z order
                    samples = value.split('x')
                    self.grid_size = (int(samples[2]), int(samples[1]), int(samples[0]))
                elif comment == '# train/val/test crop':
                    samples = value.split('x')
                    # read the crop in x, y, z order
                    self.crop_xmin = int(samples[0].split(':')[0])
                    self.crop_xmax = int(samples[0].split(':')[1])
                    self.crop_ymin = int(samples[1].split(':')[0])
                    self.crop_ymax = int(samples[1].split(':')[1])
                    self.crop_zmin = int(samples[2].split(':')[0])
                    self.crop_zmax = int(samples[2].split(':')[1])
                else:
                    sys.stderr.write('Unrecognized attribute in {}: {}\n'.format(prefix, comment))
                    sys.exit()

    def CroppingBox(self):
        if self.crop_xmin == None:
            return ((0, self.grid_size[IB_Z]), (0, self.grid_size[IB_Y]), (0, self.grid_size[IB_X]))
        return ((self.crop_zmin, self.crop_zmax), (self.crop_ymin, self.crop_ymax), (self.crop_xmin, self.crop_xmax))

    def Resolution(self):
        return self.resolution

    def GoldFilename(self):
        if self.gold_filename == None:
            return 'gold/{}_gold.h5'.format(self.prefix), 'main'
        else:
            return self.gold_filename.split()[0], self.gold_filename.split()[1]

    def SegmentationFilename(self):
        if self.segment_filename == None:
            return 'segments/{}_segment.h5'.format(self.prefix), 'main'
        else:
            return self.segment_filename.split()[0], self.segment_filename.split()[1]

    def AffinityFilename(self):
        if self.affinity_filename == None:
            return 'affinities/{}_affinities.h5'.format(self.prefix), 'main'
        else:
            return self.affinity_filename.split()[0], self.affinity_filename.split()[1]

    def GridSize(self):
        return self.grid_size

    def WriteMetaFile(self):
        meta_filename = 'meta/{}.meta'.format(self.prefix)

        with open(meta_filename, 'w') as fd:
            # write the resolution in x, y, z order
            fd.write('# resolution in nm\n')
            fd.write('{}x{}x{}\n'.format(self.resolution[2], self.resolution[1], self.resolution[0]))

            if not self.affinity_filename == None:
                fd.write('# affinity filename\n')
                fd.write('{}\n'.format(self.affinity_filename))

            fd.write('# gold filename\n')
            fd.write('{}\n'.format(self.gold_filename))

            fd.write('# segmentation filename\n')
            fd.write('{}\n'.format(self.segment_filename))

            # write the grid size in x, y, z order
            fd.write('# grid size\n')
            fd.write('{}x{}x{}\n'.format(self.grid_size[2], self.grid_size[1], self.grid_size[0]))

            if not self.crop_zmin == None:
                # write the crop in x, y, z order
                fd.write('# train/val/test crop\n')
                # need to call the function here so that we get the grid size if there was non in the original meta file
                (crop_zmin, crop_zmax), (crop_ymin, crop_ymax), (crop_xmin, crop_xmax) = self.CroppingBox()
                fd.write('{}:{}x{}:{}x{}:{}\n'.format(crop_xmin, crop_xmax, crop_ymin, crop_ymax, crop_zmin, crop_zmax))
