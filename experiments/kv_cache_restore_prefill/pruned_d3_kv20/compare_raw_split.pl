#!/usr/bin/env perl
use strict;
use warnings;
use Digest::SHA qw(sha256_hex);
use JSON::PP;

my $exp = shift // '.';

sub trim {
    my ($value) = @_;
    $value =~ s/^\s+//;
    $value =~ s/\s+$//;
    return $value;
}

sub read_all {
    my ($path) = @_;
    open my $fh, '<', $path or die "open $path: $!";
    local $/;
    my $text = <$fh>;
    close $fh;
    return $text;
}

sub raw_answer {
    my ($path) = @_;
    my $text = read_all($path);
    if ($text =~ /prompt file is [^\n]*\n/s) {
        $text = substr($text, $+[0]);
    }
    $text = (split /#################################/, $text, 2)[0];
    return trim($text);
}

sub split_answer {
    my ($path) = @_;
    my $text = (split /#################################/, read_all($path), 2)[0];
    my @lines = grep {
        $_ !~ /^config path is / &&
        $_ !~ /^CPU Group:/ &&
        $_ !~ /^The device supports:/ &&
        $_ !~ /^Loaded \d+ .*ngram table entries/ &&
        $_ !~ /^Loaded prefix-only KV cache /
    } split /\n/, $text;
    return trim(join("\n", @lines));
}

sub action_of {
    my ($text) = @_;
    my $start = index($text, '{');
    return undef if $start < 0;
    my $obj = eval { decode_json(substr($text, $start)) };
    return ref($obj) eq 'HASH' ? $obj->{action} : undef;
}

my @rows;
for my $case (0 .. 19) {
    my $id = sprintf('%02d', $case);
    my $raw = raw_answer("$exp/logs_raw_current/case_$id.log");
    my $split = split_answer("$exp/logs_split/case_$id.log");
    my $raw_action = action_of($raw);
    my $split_action = action_of($split);
    push @rows, {
        case => $id,
        output_identical => $raw eq $split ? JSON::PP::true : JSON::PP::false,
        action_identical => (defined($raw_action) && defined($split_action) && $raw_action eq $split_action)
            ? JSON::PP::true : JSON::PP::false,
        raw_action => $raw_action,
        split_action => $split_action,
        raw_sha256 => sha256_hex($raw),
        split_sha256 => sha256_hex($split),
    };
}

my $result = {
    count => scalar @rows,
    output_identical_count => scalar(grep { $_->{output_identical} } @rows),
    action_identical_count => scalar(grep { $_->{action_identical} } @rows),
    rows => \@rows,
};

open my $fh, '>', "$exp/raw_split_comparison.json" or die "write comparison: $!";
print {$fh} JSON::PP->new->canonical->pretty->encode($result);
close $fh;
print JSON::PP->new->canonical->pretty->encode($result);
