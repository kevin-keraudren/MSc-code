#!/usr/bin/python

import Manifold
import ManifoldPIL
import Utils
from glob import glob
import argparse

parser = argparse.ArgumentParser(
    usage = "usage: %(prog)s directory" )
parser.add_argument(
    'image_directory',
    metavar='image_directory',        
    help='directory of images to process' )
args = parser.parse_args()

filenames = glob(args.image_directory + '/*')

coords = Utils.read_images(filenames, 'canny')

embedded_coords, mapping = Manifold.do_embedding(coords,tree='spilltree')

ManifoldPIL.render2D([ filenames[i] for i in mapping], embedded_coords)
