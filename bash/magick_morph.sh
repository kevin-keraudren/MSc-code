#!/bin/bash
#
# Example usage:
#    ./magick_morph.sh new_image image1 image2 ... imageN
#
# input: a list of images
# output: an animated GIF between those images
#
# This script relies on ImageMagick.
#

# set -x

DELAY=20    # how much time we stop on each image
MORPH=10    # morph length
MAX_JOBS=10 # maximum number of concurrent jobs

magick_cmd="convert"

# create tmp directory
timestamp=`date +"%Y%m%d%H%M%S"`
tmp_dir=/tmp/magick_morph.${timestamp}
mkdir $tmp_dir

i=2
while [ $i -le $# ]
do
    prefix=${!i##*.}
    base=`basename ${!i} .$prefix`
    magick_cmd="${magick_cmd} ${tmp_dir}/${base}.gif"
    ( convert ${!i} -delay $DELAY ${tmp_dir}/${base}.gif && touch ${tmp_dir}/${base}.done ) &
    j=$((i+1))
    if [ $j -le $# ]
    then
        ( convert ${!i} ${!j} -morph $MORPH ${tmp_dir}/${i}-${j}.gif && touch ${tmp_dir}/${i}-${j}.done ) &
        magick_cmd="${magick_cmd} ${tmp_dir}/${i}-${j}.gif"
    fi
    i=$((i+1))

    joblist=($(jobs -p))
    while (( ${#joblist[*]} >= MAX_JOBS ))
    do
        sleep 1
        joblist=($(jobs -p))
    done
    
done

magick_cmd="${magick_cmd} -loop 0 $1"

nb_created=$[ 2 * ( $# - 1 ) - 1 ]
nb_current=`ls ${tmp_dir}/*.done | wc -l`
while [ $nb_current -lt $nb_created ]
do
    sleep 1
    nb_current=`ls ${tmp_dir}/*.done | wc -l`
done

$magick_cmd

name=${1%%.*}
convert $1 -colorspace gray ${name}_gray.gif

rm -r $tmp_dir
