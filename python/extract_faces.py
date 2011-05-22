#!/usr/bin/python
"""
Adapted from OpenCV Python sample and code from the FaceCrumbs project
"""

import sys
import os
import argparse

import cv

def detect_and_crop(image_file, cascade, crop_size,
                    output_dir=None,unique=False, verbose=False):

    (basename,prefix) = os.path.splitext(image_file)

    img = cv.LoadImage( image_file, 1)

    if output_dir:
        basename = output_dir + '/' + os.path.basename(basename)
    
    # allocate temporary images
    gray = cv.CreateImage((img.width,img.height), 8, 1)

    # convert color input image to grayscale
    cv.CvtColor(img, gray, cv.CV_BGR2GRAY)

    # equalize histogram
    cv.EqualizeHist(gray, gray)

    faces = cv.HaarDetectObjects( img,
                                  cascade,
                                  cv.CreateMemStorage(0),
          # The factor by which the search window is scaled between the 
          # subsequent scans, 1.1 means increasing window by 10%
                                  1.2,
          # Minimum number (minus 1) of neighbor rectangles that makes 
          # up an object
                                  3,
          # CV_HAAR_DO_CANNY_PRUNNING
                                  1,
          # Minimum window size
                                  (40,40) ) # minimum size


    if faces and (len(faces)==1 or not unique):
        count = 0
        for ((x, y, w, h), n) in faces:
            if verbose: print ((x, y, w, h), n)

            cv.SetImageROI(img, (x, y, w, h));

            # create destination image
            # Note that cvGetSize will return the width and the height of ROI
            crop = cv.CreateImage((w,h), img.depth, img.nChannels)

            resized = cv.CreateImage( crop_size, cv.IPL_DEPTH_8U, img.nChannels)
    
            # copy subimage
            cv.Copy(img, crop)
            cv.Resize(crop, resized)

            cv.SaveImage(basename + '_' + str(count) +'.jpg', resized)
            count += 1

            # always reset the Region of Interest
            cv.ResetImageROI(img)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        usage = "usage: %(prog)s [options] [filename]" )
    parser.add_argument(
        'images',
        metavar='images',
        nargs='+',
        help='images to process' )
    parser.add_argument(
        "-c",
        "--cascade",
        action="store",
        dest="cascade",
        help="Haar cascade file, default %(default)s",
        default =
        "/usr/share/doc/opencv-doc/examples/haarcascades/haarcascades/haarcascade_frontalface_alt.xml.gz" )
    parser.add_argument(
        '--crop',
        nargs=2,
        default=(64,64),
        type=int,
        help="crop size" )
    parser.add_argument(
        '--output_dir',
        help="output directory (default to input directory)" )
    parser.add_argument(
        '--unique',
        action="store_true", default=False,
        help="disregard results when multiple faces detected" )
    parser.add_argument(
        '--verbose',
        action="store_true", default=False,
        help="verbose mode" ) 
    
    args = parser.parse_args()

    cascade = cv.Load(args.cascade)

    for image_file in args.images:
        detect_and_crop(image_file, cascade, args.crop,
        output_dir=args.output_dir, unique=args.unique, verbose=args.verbose)

