#!/usr/bin/perl
#
# Example usage:
#    ./convert_OS_postcodes_to_LatLong.pl *.csv
#
# input: Ordnance Survey files "Code-Point Open/data/CSV/*.csv"
# output: list of postcodes with lat-long coordinates
#

use strict;
use warnings;

use Geo::Coordinates::OSGB 'grid_to_ll';

while (<>) {
    my @line = split( ',', $_);
    my $postcode = $line[0];
    my $easting  = $line[10];
    my $northing = $line[11];

    next unless ( $postcode
		  && $easting
		  && $northing );

    $postcode =~ s/"//g;

    my ( $lat, $lon ) = grid_to_ll( $easting, $northing );

    print $postcode . "\t" 
	. $lat . ',' . $lon
	. "\n";  
}

exit(0);
