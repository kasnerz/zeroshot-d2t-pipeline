#!/usr/bin/env python3

import json
import os
import re
import logging
import argparse

from collections import defaultdict, deque

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO, datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

def are_buckets_full(buckets, bucket_size):
    return all([len(bucket) >= bucket_size for bucket in buckets])


def skip_paragraph(paragraph, args):
    filter_out_patterns = [
        r"\([^w]*;", # broken information in parentheses
        r"\(\s*,",
        r",\s*\)",
        r"\s+\)",
        r"\(\s*\)",
        r",\s*,",    # broken punctuation
        r"\s+;",
        r"\s+\.",
        r"[^\.]$",   # wrong ending, e.g. "can refer to:"
        r"^[\s,.?!]", # punctuation in the beginning of the paragraph
        r"^This is",  # not a regular article 
        r"^These are",
        r"^A list of",
        r"^The following is",
        r"^Statistics of"  
    ] 
    if len(paragraph) >= args.max_len or len(paragraph) < args.min_len:
        return True

    for pattern in filter_out_patterns:
        if re.search(pattern, paragraph) is not None:
            return True

    return False


def filter_paragraphs(args, buckets):
    ctrs = defaultdict(int)

    beginnings = set()
    bows = deque()
    bucket_range = (args.max_len - args.min_len) // args.buckets

    for root, subdirs, files in os.walk(args.input):
        for file in files:
            path = os.path.join(root, file)
            logger.info(f"Processing {path}")

            with open(path) as f:
                for line in f.readlines():
                    article = json.loads(line)

                    if not article["text"]:
                        ctrs["missing_text"] += 1
                        continue

                    first_paragraph = article["text"].split("\n")[0]

                    if skip_paragraph(first_paragraph, args):
                        ctrs["skipped"] += 1
                        continue

                    par_len = len(first_paragraph)
                    bucket_idx = (par_len - args.min_len) // bucket_range

                    if len(buckets[bucket_idx]) >= args.bucket_size:
                        ctrs["full_bucket"] += 1
                        continue

                    if first_paragraph[:args.skip_beg_chars] in beginnings:
                        ctrs["same_beginning"] += 1
                        continue
                    beginnings.add(first_paragraph[:args.skip_beg_chars])


                    bow = set(first_paragraph.split())
                    is_repeated = False
                    for prev_bow in bows:
                        if len(prev_bow & bow) > args.bow_difference * len(bow):
                            is_repeated = True
                            break

                    if is_repeated:
                        ctrs["repeated_words"] += 1
                        continue

                    bows.append(bow)

                    if len(bows) > args.bow_keep:
                        bows.popleft()

                    if args.remove_years:
                        first_paragraph = re.sub(r"\([^\)]*\d{4}[^\)]*\) ", "", first_paragraph)

                    buckets[bucket_idx].append(first_paragraph)
                    ctrs["correct"] += 1

                    if are_buckets_full(buckets, args.bucket_size):
                        return

            logger.info([len(bucket) for bucket in buckets])
            logger.info(ctrs)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="en-wiki", type=str,
        help="Path to the output directory from wikiextractor (containing subdirs AA, AB, ...).")
    parser.add_argument("--output", default="wiki_paragraphs_filtered.txt", type=str,
        help="Output file.")
    parser.add_argument("--buckets", default=4, type=int,
        help="Number of buckets.")
    parser.add_argument("--bucket_size", default=250000, type=int,
        help="Sentences per bucket.")
    parser.add_argument("--min_len", default=30, type=int,
        help="Minimum length of the paragraph.")
    parser.add_argument("--max_len", default=430, type=int,
        help="Maximum length of the paragraph.")
    parser.add_argument("--skip_beg_chars", default=15, type=int,
        help="Skip all paragraphs which first skip_beg_chars characters are identical to some already extracted paragraph \
         (useful to skip meta-articles etc.).")
    parser.add_argument("--bow_difference", default=0.75, type=float,
        help="Skip all paragraph which contain bow_difference * 100% words identical to one of the last bow_keep paragraph.")
    parser.add_argument("--bow_keep", default=500, type=int,
        help="Keep last bow_keep paragraphs for comparisson and filtering (larger numbers can slow down the extraction process).")
    parser.add_argument("--remove_years", action="store_true",
        help="Delete a very common pattern of years in parentheses.")

    args = parser.parse_args()
    logger.info(args)

    buckets = [[] for i in range(args.buckets)]

    filter_paragraphs(args, buckets)

    with open(args.output, "w") as out_file:
        for bucket in buckets:
            for sent in bucket:
                out_file.write(sent + "\n")
