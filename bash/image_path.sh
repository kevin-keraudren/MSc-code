#!/bin/bash
#
# Example usage:
#     ./image_path.sh new_image image1 image2 ... imageN
#
# input: a list of images
# output: all the images concatenated on a single line
#
# This script relies on ImageMagick.
#


# set -x

# for more examples, see: http://www.imagemagick.org/Usage/montage/

montage ${@:2} -tile x1 -geometry +0+0 $1

name=${1%%.*}
convert $1 -colorspace gray ${name}_gray.png

# montage ${@:2} -tile x1 -shadow -geometry +1+1 -background none $1

# montage ${@:2} -tile x1 -shadow -geometry +1+1 -background white $1

exit 0

