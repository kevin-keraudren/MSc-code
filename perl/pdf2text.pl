#!/usr/bin/perl
#
# Example usage:
#    ./pdf2text.pl old.pdf new.txt
#    ./pdf2text.pl old.pdf > new.txt
#
# input: PDF file
# output: text with spaces simplified and non-ASCII characters removed
#
# This script rely on the Linux program "pdftotext".
#

use strict;
use warnings;

my $infile = $ARGV[0];
my $outfile = $ARGV[1];

# convert PDF to text
my $text = `pdftotext -nopgbrk -eol unix "$infile" -`;

# do some cleaning
$text =~ s/[^[:ascii:]]+/ /g;
$text =~ s/ +/ /g; # \s also matchs \n

# output
if ( $outfile ) {
    open( my $fh, '>', $outfile);
    print $fh $text;
    close($fh);
}
else {
    print $text;
}

exit(0);


