import json
import math
import os
import pickle
import sys
from typing import Dict, List


class BM25:
    EPSILON = 0.25
    PARAM_K1 = 1.5
    PARAM_B = 0.6

    def __init__(self, corpus: Dict):
        self.corpus_size = 0
        self.wordNumsOfAllDoc = 0
        self.doc_freqs = {}
        self.idf = {}
        self.doc_len = {}
        self.docContainedWord = {}
        self._initialize(corpus)

    def _initialize(self, corpus: Dict):
        for index, document in corpus.items():
            self.corpus_size += 1
            self.doc_len[index] = len(document)
            self.wordNumsOfAllDoc += len(document)

            frequencies = {}
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            self.doc_freqs[index] = frequencies

            for word in frequencies.keys():
                if word not in self.docContainedWord:
                    self.docContainedWord[word] = set()
                self.docContainedWord[word].add(index)

        idf_sum = 0
        negative_idfs = []
        for word in self.docContainedWord.keys():
            doc_nums_contained_word = len(self.docContainedWord[word])
            idf = math.log(self.corpus_size - doc_nums_contained_word + 0.5) - math.log(doc_nums_contained_word + 0.5)
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)
        average_idf = float(idf_sum) / len(self.idf)
        eps = BM25.EPSILON * average_idf
        for word in negative_idfs:
            self.idf[word] = eps

    @property
    def avgdl(self):
        return float(self.wordNumsOfAllDoc) / self.corpus_size

    def get_score(self, query: List, doc_index):
        k1 = BM25.PARAM_K1
        b = BM25.PARAM_B
        score = 0
        doc_freqs = self.doc_freqs[doc_index]
        for word in query:
            if word not in doc_freqs:
                continue
            score += self.idf[word] * doc_freqs[word] * (k1 + 1) / (
                        doc_freqs[word] + k1 * (1 - b + b * self.doc_len[doc_index] / self.avgdl))
        return [doc_index,score]
    def get_scores(self,query):
        scores = [self.get_score(query,index) for index in self.doc_len.keys()]
        return scores