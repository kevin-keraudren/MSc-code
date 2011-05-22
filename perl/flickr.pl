#!/usr/bin/perl
#
# Written by Kevin Keraudren, 07/11/2010
# kevin.keraudren10@imperial.ac.uk
#
# See usage() below, or run "./flickr.pl -h" to see how to use this script.
#

use strict;
use warnings;
use English;

use Flickr::API;
use Getopt::Long;
use LWP::Simple;

use Data::Dumper;

use Parallel::ForkManager;

my $MAX_PROCESSES = 100;

my $api_key = ''; # Put your API key here
my $api_secret = ''; # Put your API secret here

my $tags;
my $bbox; # min_long,min_lat,max_long,max_lat
my $verbose = 1;
my $help;
my $dir;
my $size = 'o';
my $options = GetOptions (
    "size=s"   => \$size,
    "dir=s"    => \$dir,
    "tags=s"   => \$tags,
    "bbox=s"   => \$bbox,
    "verbose"  => \$verbose,
    "help"     => \$help,
    );

unless ( $dir && ( $tags || $bbox ) ) {
    usage();
}
if ( $size !~ /^[tmzbo]$/) {
    print "**ERROR**: Unknown size '$size'\n";
    usage();
}
unless ( -e $dir ) {
    die "**ERROR**: '$dir' does not exist";
}
if ( $help ) {
    usage();
}

my $api = new Flickr::API( { key    => $api_key,
                             secret => $api_secret,
                           } );

my $search_options = {
    tags => $tags,
    bbox => $bbox,
};
$search_options->{extras} = 'url_o' if $size eq 'o';

my $response = $api->execute_method( 'flickr.photos.search', $search_options );

my @photos;
foreach my $item ( @{$response->{tree}->{children}->[1]->{children}}) {
    if ($item->{attributes} ){
        if ( $size ne 'o' || $item->{attributes}->{url_o} ) {
            push( @photos, $item->{attributes} );
        }
    }
}
print "Found " . scalar( @photos ) . " photos\n" if $verbose;
# print Dumper(\@photos);die;

my $pm = new Parallel::ForkManager($MAX_PROCESSES);

foreach my $photo ( @photos ) {
    my $pid = $pm->start() and next;
    # SIZE:
    #     t	thumbnail, 100 on longest side
    #     m	small, 240 on longest side
    #     z	medium 640, 640 on longest side
    #     b	large, 1024 on longest side

    my $basename = $photo->{id}
    . '_' . $photo->{secret}
    . '_' . $size . '.jpg'; # size

    my $url;

    if ( $size eq 'o' ) {
        $url = $photo->{url_o};
    }
    else {
        $url = 'http://farm' . $photo->{farm}
        . '.static.flickr.com/'. $photo->{server}
        . '/' . $basename;
    }

    my $file = "$dir/$basename";
    getstore( $url, $file );
    #print "( wget -q -t 1 -T 1 -O '$file' '$url' ) &\n";
    #system("( wget -q -O '$file' '$url' ) &"); # we are not waiting for the download to finish
    print "$url\n" if $verbose;
    $pm->finish(); # Terminates the child process
}

$pm->wait_all_children();

exit(0);

sub usage {
    my $usage =<<END

    This script uses Flickr::API to download images from Flickr, either
    searching with a bounding box (--bbox) and/or tags (--tags).
    It was originally written to download the original images, so that they can
    be used in the Bundler pipeline
    (http://phototour.cs.washington.edu/bundler/) thanks to their EXIF tags, but
    it can be used to download any size of images (default to original).

    You need to hardcode your own API key and secret at the beginning of this
    script to use the Flickr API.

    Using both tags and bbox is strongly recommended to get relevant results.
    
    Usage: $PROGRAM_NAME --tags coliseum --dir coliseum --bbox 12.492,41.890,12.493,41.891

    Options:
        --size     t thumbnail, m small, z medium, b large, o orginial (default
                   to o)
        --dir      directory where to download the photos
        --tags     comma separated list of tags
        --bbox     min_long,min_lat,max_long,max_lat
        --verbose  print some debugging information, default to 1
        --help     print this message
    
END
;
    print $usage;
    exit(1);
}
