#!/usr/bin/env python3

import sys

with open(sys.argv[1]) as f:
    lines = f.readlines()
    N = len(lines)

    total_omissions = 0
    total_hallucinations = 0
    total_templates = 0

    for line in lines:
        template_count, omissions, hallucinations = [int(x) for x in line.split()]

        total_templates += template_count
        total_omissions += omissions
        total_hallucinations += hallucinations

    print(f"o: {total_omissions} h: {total_hallucinations}")
    print(f"or: {total_omissions/total_templates:.3f}, oer: {total_omissions/N:.3f}, hr: {total_hallucinations/N:.3f}")