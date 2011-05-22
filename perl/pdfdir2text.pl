#!/usr/bin/perl
#
# Example usage:
#    ./pdfdir2text.pl dir1 dir2 ... dirN new_dir
#
# input: list of directories
# output: the same directory hierarchy is recreated inside new_dir, and all PDF
# files have been turned into text files.
#
# This script relies on "pdf2text.pl", which itself rely on the Linux program
# "pdftotext".
#

use strict;
use warnings;

use File::Find;
use File::Basename;
use File::Path qw(make_path);

my @input_directories = @ARGV[ 0 .. scalar(@ARGV)-2 ];
my $output_directory = $ARGV[-1];

my @pdfs;

sub wanted {
    if ( /\.pdf$/ ) {
        push( @pdfs, $File::Find::name );
    }
}

find( \&wanted, @ARGV[ 0 .. scalar(@ARGV)-2 ] );

foreach my $pdf ( @pdfs ) {
    my $dir = dirname( $pdf );
    my $out_dir = $output_directory . '/' . $dir;
    unless ( -d $out_dir ) {
        make_path( $out_dir );
    }
    my $txt = basename( $pdf, '.pdf' );
    print "Converting '$pdf'\n";
    print `pdf2text.pl "$pdf" "$out_dir/$txt.txt"`;
}

print "Converted " . scalar(@pdfs) . " pdf files\n";

exit(0);


