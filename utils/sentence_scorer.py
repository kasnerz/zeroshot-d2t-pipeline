#!/usr/bin/env python

import logging

from lm_scorer.models.auto import AutoLMScorer as LMScorer
from pprint import pprint as pp
from utils.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


class SentenceScorer:
    def __init__(self, reduce_mode="gmean", device="cuda"):
        if device == "cpu":
            logger.warning("Running LMScorer on CPU. Scoring may be slow.")

        self.model = LMScorer.from_pretrained("distilgpt2", device=device, batch_size=1)
        self.reduce_mode = reduce_mode
        self.tokenizer = Tokenizer()

    def score(self, sentence):
        sentence = self.tokenizer.detokenize(sentence)

        return self.model.sentence_score(sentence, reduce=self.reduce_mode, log=True)

    def select_best(self, sentences):
        scores = []

        for sent in sentences:
            sent_score = self.score(sent)
            scores.append((sent, sent_score))

        scores.sort(key=lambda x: x[1], reverse=True)
        # pp(scores)

        return scores[0][0]