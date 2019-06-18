import os
import numpy as np

from ibex.utilities import dataIO



def ImageStackToH5(image_directory, output_filename):   
    # get all of the image file
    image_filenames = sorted(os.listdir(image_directory))

    zres = len(image_filenames)
    yres, xres = dataIO.ReadImage('{}/{}'.format(image_directory, image_filenames[0])).shape
    image = np.zeros((zres, yres, xres), dtype=np.uint8)

    # get the number of images in the directory
    for iz, image_filename in enumerate(image_filenames):
        print image_filename

        # read the image file
        filename = '{}/{}'.format(image_directory, image_filename)
        image_slice = dataIO.ReadImage(filename)
        image[iz,:,:] = image_slice

    dataIO.WriteH5File(image, output_filename, 'main')