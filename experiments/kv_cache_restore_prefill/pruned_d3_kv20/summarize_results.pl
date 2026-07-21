#!/usr/bin/env perl
use strict;
use warnings;
use Digest::SHA qw(sha256_hex);
use JSON::PP;

my $exp = shift // ".";
my @rows;

sub parse_log {
    my ($path, $prefix_label) = @_;
    open my $fh, '<', $path or die "open $path: $!";
    local $/;
    my $text = <$fh>;
    close $fh;

    my ($answer_part) = split /#################################/, $text, 2;
    my @answer_lines = grep {
        $_ !~ /^config path is / &&
        $_ !~ /^CPU Group:/ &&
        $_ !~ /^The device supports:/ &&
        $_ !~ /^Loaded \d+ .*ngram table entries/ &&
        $_ !~ /^Loaded prefix-only KV cache /
    } split /\n/, $answer_part;
    my $answer = join("\n", @answer_lines) . "\n";

    my ($prefix_tokens) = $text =~ /\Q$prefix_label\E = (\d+)/;
    my ($variable_tokens) = $text =~ /variable tokens num = (\d+)/;
    my ($decode_tokens) = $text =~ /decode tokens num = (\d+)/;
    my ($prefill_s) = $text =~ /prefill time = ([0-9.]+) s/;
    my ($decode_s) = $text =~ /decode time = ([0-9.]+) s/;
    die "missing metrics in $path" unless defined $prefix_tokens && defined $variable_tokens &&
        defined $decode_tokens && defined $prefill_s && defined $decode_s;

    return {
        answer => $answer,
        answer_sha256 => sha256_hex($answer),
        prefix_tokens => 0 + $prefix_tokens,
        variable_tokens => 0 + $variable_tokens,
        decode_tokens => 0 + $decode_tokens,
        prefill_s => 0 + $prefill_s,
        decode_s => 0 + $decode_s,
    };
}

for my $case (0 .. 19) {
    my $id = sprintf('%02d', $case);
    my $split = parse_log("$exp/logs_split/case_$id.log", 'split prefix tokens num');
    my $restore = parse_log("$exp/logs_restore/case_$id.log", 'cached prefix tokens num');
    push @rows, {
        case => $id,
        prefix_variant => ($id =~ /^(08|14|17|19)$/ ? 'B' : 'A'),
        prefix_tokens => $split->{prefix_tokens},
        variable_tokens => $split->{variable_tokens},
        decode_tokens => $split->{decode_tokens},
        split_prefill_s => $split->{prefill_s},
        restore_prefill_s => $restore->{prefill_s},
        prefill_speedup => $restore->{prefill_s} > 0 ? $split->{prefill_s} / $restore->{prefill_s} : 0,
        split_decode_s => $split->{decode_s},
        restore_decode_s => $restore->{decode_s},
        split_sha256 => $split->{answer_sha256},
        restore_sha256 => $restore->{answer_sha256},
        output_identical => $split->{answer} eq $restore->{answer} ? JSON::PP::true : JSON::PP::false,
    };
}

my $sum = sub {
    my ($field) = @_;
    my $total = 0;
    $total += $_->{$field} for @rows;
    return $total;
};

my $count = scalar @rows;
my $split_prefill = $sum->('split_prefill_s');
my $restore_prefill = $sum->('restore_prefill_s');
my $split_decode = $sum->('split_decode_s');
my $restore_decode = $sum->('restore_decode_s');
my $decode_tokens = $sum->('decode_tokens');
my $identical_count = scalar grep { $_->{output_identical} } @rows;
my @prefill_speedups = sort { $a <=> $b } map { $_->{prefill_speedup} } @rows;
my $prefill_speedup_median = ($prefill_speedups[9] + $prefill_speedups[10]) / 2;
my $prefill_improved_count = scalar grep { $_ > 1 } @prefill_speedups;

my $summary = {
    count => $count,
    output_identical_count => $identical_count,
    output_identical_rate => $identical_count / $count,
    avg_prefix_tokens => $sum->('prefix_tokens') / $count,
    avg_variable_tokens => $sum->('variable_tokens') / $count,
    avg_decode_tokens => $decode_tokens / $count,
    split_prefill_total_s => $split_prefill,
    restore_prefill_total_s => $restore_prefill,
    split_prefill_avg_s => $split_prefill / $count,
    restore_prefill_avg_s => $restore_prefill / $count,
    prefill_speedup => $split_prefill / $restore_prefill,
    prefill_speedup_min => $prefill_speedups[0],
    prefill_speedup_median => $prefill_speedup_median,
    prefill_speedup_max => $prefill_speedups[-1],
    prefill_improved_count => $prefill_improved_count,
    prefill_reduction => 1 - $restore_prefill / $split_prefill,
    split_decode_total_s => $split_decode,
    restore_decode_total_s => $restore_decode,
    split_decode_tok_s => $decode_tokens / $split_decode,
    restore_decode_tok_s => $decode_tokens / $restore_decode,
    split_prefill_decode_total_s => $split_prefill + $split_decode,
    restore_prefill_decode_total_s => $restore_prefill + $restore_decode,
    prefill_decode_speedup => ($split_prefill + $split_decode) / ($restore_prefill + $restore_decode),
    prefill_decode_reduction => 1 - ($restore_prefill + $restore_decode) / ($split_prefill + $split_decode),
};

open my $json_fh, '>', "$exp/results.json" or die "write results.json: $!";
print {$json_fh} JSON::PP->new->canonical->pretty->encode({summary => $summary, cases => \@rows});
close $json_fh;

open my $csv_fh, '>', "$exp/results.csv" or die "write results.csv: $!";
print {$csv_fh} "case,prefix_variant,prefix_tokens,variable_tokens,decode_tokens,split_prefill_s,restore_prefill_s,prefill_speedup,split_decode_s,restore_decode_s,output_identical\n";
for my $row (@rows) {
    print {$csv_fh} join(',', map { $row->{$_} } qw(
        case prefix_variant prefix_tokens variable_tokens decode_tokens split_prefill_s
        restore_prefill_s prefill_speedup split_decode_s restore_decode_s output_identical
    )), "\n";
}
close $csv_fh;

print JSON::PP->new->canonical->pretty->encode($summary);
