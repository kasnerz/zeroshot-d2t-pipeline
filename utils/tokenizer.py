#!/usr/bin/env python

from sacremoses import MosesDetokenizer
from nltk import word_tokenize, sent_tokenize
import re

class Tokenizer:
    def __init__(self):
        self.detokenizer = MosesDetokenizer(lang='en')

    def normalize(self, s, remove_quotes=False, remove_parentheses=False):
        # remove underscores
        s = re.sub(r'_', r' ', s)

        if remove_quotes:
            s = re.sub(r'"', r'', s)
            s = re.sub(r'``', r'', s)
            s = re.sub(r"''", r'', s)

        if remove_parentheses:
            s = re.sub(r'\(', r'', s)
            s = re.sub(r'\)', r'', s)

        # split basic camel case, lowercase first letters
        s = re.sub(r"([a-z])([A-Z])",
            lambda m: rf"{m.group(1)} {m.group(2).lower()}", s)

        return s
        

    def tokenize(self, s):
        tokens = []

        for sentence in sent_tokenize(s):
            s = self.normalize(s)

            # NLTK word tokenize
            tokens += word_tokenize(sentence)

        res = " ".join(tokens)
        return res


    def detokenize(self, s):
        tokens = s.split()

        return self.detokenizer.detokenize(tokens)