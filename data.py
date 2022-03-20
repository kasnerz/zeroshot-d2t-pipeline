#!/usr/bin/env python3

import json
import csv
import os
import logging
import re
import random

from collections import defaultdict, namedtuple
from utils import webnlg_parsing
from utils.corpus_reader.benchmark_reader import Benchmark, select_files
from utils.tokenizer import Tokenizer

logger = logging.getLogger(__name__)

DataTriple = namedtuple('DataTriple', ['subj', 'pred', 'obj'])

def get_dataset_class(dataset_class):
    """
    A wrapper for easier introduction of new datasets.
    Returns class "MyDataset" for a parameter "--dataset mydataset"
    """
    try:
        # case-insensitive
        available_classes = {o.name.lower() : o for o in globals().values() 
                                if type(o)==type(D2TDataset) and hasattr(o, "name")}
        return available_classes[dataset_class.lower()]
    except AttributeError:
        logger.error(f"Unknown dataset: '{args.dataset}'. Please create \
            a class with an attribute name='{args.dataset}' in 'data.py'.")
        return None

class DataEntry:
    """
    A single D2T dataset example: a set of triples & its possible lexicalizations
    """
    def __init__(self, triples, lexs):
        self.triples = triples
        self.lexs = lexs

    def __repr__(self):
        return str(self.__dict__)

class D2TDataset:
    def __init__(self):
        self.data = {split: [] for split in ["train", "dev", "test"]}
        self.fallback_template = "The <predicate> of <subject> is <object> ."

    def load_from_dir(self, path, template_path, splits):
        """
        Load the dataset
        """
        raise NotImplementedError

    def load_templates(self, templates_filename):
        """
        Load existing templates from a JSON file
        """
        logger.info(f"Loaded templates from {templates_filename}")
        with open(templates_filename) as f:
            self.templates = json.load(f)


class WebNLG(D2TDataset):
    name="webnlg"

    def __init__(self):
        super().__init__()

    def get_template(self, triple):
        """
        Return the template for the triple
        """
        pred = triple.pred

        if pred in self.templates:
            # Using just a single template
            assert len(self.templates[pred]) == 1
            template = self.templates[pred][0]
        else:
            logger.warning(f"No template for {pred}, using a fallback")
            template = self.fallback_template

        return template

    def load_from_dir(self, path, template_path, splits):
        """
        Load the dataset
        """
        self.load_templates(template_path)

        for split in splits:
            logger.info(f"Loading {split} split")
            data_dir = os.path.join(path, split)
            err = 0
            xml_entryset = webnlg_parsing.run_parser(data_dir)

            for xml_entry in xml_entryset:
                triples = [DataTriple(e.subject, e.predicate, e.object) 
                                for e in xml_entry.modifiedtripleset]
                lexs = self._extract_lexs(xml_entry.lexEntries, triples)

                if not any([lex for lex in lexs]):
                    err += 1
                    continue

                entry = DataEntry(triples=triples, lexs=lexs)
                self.data[split].append(entry)

            if err > 0:
                logger.warning(f"Skipping {err} entries without lexicalizations...")

    def _extract_lexs(self, lex_entries, triples):
        """
        Use `orderedtripleset` in the WebNLG dataset to determine the "ground-truth" order
        of the triples (based on human references).
        """
        lexs = []

        for entry in lex_entries:
            order, agg = self._extract_ord_agg(triples, entry.orderedtripleset)
            lex = {
                "text" : entry.text,
                "order" : order,
                "agg" : agg
            }
            lexs.append(lex)

        return lexs

    def _extract_ord_agg(self, triples, ordered_triples):
        """
        Determine the permutation indices and aggregation markers from
        the ground truth.
        """
        # if ordered triples do not match the actual triples -> fail
        ordered_triples_flattened = [x for sent in ordered_triples for x in sent]
        if len(ordered_triples_flattened) != len(triples):
            return None, None

        order = []

        for t in triples:
            for i, o in enumerate(ordered_triples_flattened):
                if t.subj == o.subject and \
                   t.pred == o.predicate and \
                   t.obj == o.object:
                   order.append(i)
                   break
            else:
                # ordered triples do not match the actual triples
                return None, None

        agg = []

        for i, triples_in_sent in enumerate(ordered_triples):
            if triples_in_sent:
                agg += [i] * len(triples_in_sent)

        return order, agg



class E2E(D2TDataset):
    name="e2e"

    def __init__(self):
        super().__init__()

    def load_from_dir(self, path, template_path, splits):
        """
        Load the dataset
        """
        self.load_templates(template_path)

        for split in splits:
            logger.info(f"Loading {split} split")
            triples_to_lex = defaultdict(list)

            with open(os.path.join(path, f"{split}.csv")) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',', quotechar='"')

                # skip header
                next(csv_reader)
                err = 0

                for i, line in enumerate(csv_reader):
                    triples = self._mr_to_triples(line[0])

                    # probably a corrupted sample
                    if not triples or len(triples) == 1:
                        err += 1
                        # cannot skip for dev and test
                        if split == "train":
                            continue

                    lex = {"text" : line[1]}
                    triples_to_lex[triples].append(lex)

                # triples are not sorted, complete entries can be created only after the dataset is processed
                for triples, lex_list in triples_to_lex.items():
                    entry = DataEntry(triples, lex_list)
                    self.data[split].append(entry)

            logger.warn(f"{err} corrupted instances")


    def _mr_to_triples(self, mr):
        """
        Transforms E2E meaning representation into RDF triples.
        """
        triples = []

        # cannot be dictionary, slot keys can be duplicated
        items = [x.strip() for x in mr.split(",")]
        subj = None

        keys = []
        vals = []

        for item in items:
            key, val = item.split("[")
            val = val[:-1]

            keys.append(key)
            vals.append(val)

        name_idx = None if "name" not in keys else keys.index("name")
        eatType_idx = None if "eatType" not in keys else keys.index("eatType")

        # primary option: use `name` as a subject
        if name_idx is not None:
            subj = vals[name_idx]
            del keys[name_idx]
            del vals[name_idx]

            # corrupted case hotfix
            if not keys:
                keys.append("eatType")
                vals.append("restaurant")

        # in some cases, that does not work -> use `eatType` as a subject
        elif eatType_idx is not None:
            subj = vals[eatType_idx]
            del keys[eatType_idx]
            del vals[eatType_idx]
        # still in some cases, there is not even an eatType 
        #-> hotfix so that we do not lose data
        else:
            # logger.warning(f"Cannot recognize subject in mr: {mr}")
            subj = "restaurant"

        for key, val in zip(keys, vals):
            triples.append(DataTriple(subj, key, val))

        # will be used as a key in a dictionary
        return tuple(triples)


    def get_template(self, triple):
        """
        Return the template for the triple
        """
        if triple.pred in self.templates:
            templates = self.templates[triple.pred]
            # special templates for familyFriendly yes / no
            if type(templates) is dict and triple.obj in templates:
                template = templates[triple.obj][0]
            else:
                template = templates[0]
        else:
            template = self.fallback_template

        return template

