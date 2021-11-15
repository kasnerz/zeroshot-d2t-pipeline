#!/usr/bin/env python3


from slot_error import process_file
import sys
import csv

split="test"
input_file = sys.argv[1]
mr_file = f"data/e2e_mrs/{split}.ref"
output_file = input_file + ".csv"

with open(input_file) as f_in, \
     open(mr_file) as f_mr, \
     open(output_file, "w") as f_out:
    lines_in = f_in.readlines()
    lines_mr = f_mr.readlines()

    assert(len(lines_in) == len(lines_mr))

    header = "mr,ref\n"
    f_out.write(header)

    for mr, line in zip(lines_mr, lines_in):
        mr = mr.rstrip("\n")
        line = line.rstrip("\n")
        f_out.write(f'"{mr}","{line}"\n')

process_file(output_file)