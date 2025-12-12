# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""This module manages all the metrics occurring at the corpus level.
Some metrics (such as corpus BLEU) are not computed at the individual item level, but over all the corpus.
A number of these aggregations come from the EleutherAIHarness
"""

import logging
import math
from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
import sacrebleu
import sklearn.metrics

from lighteval.metrics.sample_preparator import (
    GenerativeCorpusMetricInput,
    LogprobCorpusMetricInput,
    PerplexityCorpusMetricInput,
)
from lighteval.utils.utils import as_list


logger = logging.getLogger(__name__)


class CorpusLevelComputation(ABC):
    @abstractmethod
    def compute_corpus(self, items):
        raise NotImplementedError

    def __str__(self):
        attrs = vars(self)
        attr_strs = []
        for k, v in attrs.items():
            if callable(v):
                val_str = v.__name__
            else:
                val_str = str(v)
            attr_strs.append(f"{k}={val_str}")
        return f"{self.__class__.__name__}({', '.join(attr_strs)})"


# General aggregations
class MatthewsCorrCoef(CorpusLevelComputation):
    def compute_corpus(self, items: list[GenerativeCorpusMetricInput]) -> float:
        """Computes the Matthews Correlation Coefficient, using scikit learn ([doc](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html)).

        Args:
            items (list[dict]): List of GenerativeCorpusMetricInput

        Returns:
            float: Score
        """
        golds = [i.golds for i in items]
        preds = [i.preds for i in items]
        return sklearn.metrics.matthews_corrcoef(golds, preds)


class CorpusLevelF1Score(CorpusLevelComputation):
    def __init__(self, average: str, num_classes: int = 2):
        """Stores the relevant parameters for the task's corpus level f1 score.

        Args:
            average (str): Method to use to compute the f1 score. Can be weighted, macro, micro.
            num_classes (int, optional): Num of possible choice classes. Defaults to 2. If this parameter is above 2, we'll compute multi f1 corpus score
        """
        if average not in ["weighted", "macro", "micro", None]:
            raise ValueError(
                f"A CorpusLevelF1Score must be initialized with weighted, macro, micro, or None as an average function. {average} was used."
            )
        self.average = average
        self.num_classes = num_classes

    def compute_corpus(self, items: list[LogprobCorpusMetricInput]):
        """Computes the metric score over all the corpus generated items, by using the scikit learn implementation."""
        golds = [i.golds for i in items]
        preds = [i.preds for i in items]
        # Single f1
        if self.num_classes == 2:
            fscore = sklearn.metrics.f1_score(golds, preds, average=self.average)
            return np.max(fscore)

        # Multi f1
        f1s = []
        for i in range(self.num_classes):
            f1s.append(
                sklearn.metrics.f1_score(
                    y_true=[g == i for g in golds], y_pred=[p == i for p in preds], average=self.average
                )
            )
        return float(np.mean(f1s))


class CorpusLevelTranslationMetric(CorpusLevelComputation):
    def __init__(self, metric_type: str, lang: Literal["zh", "ja", "ko", ""] = ""):
        """Stores the relevant parameters for a corpus level translation metric.

        Args:
            metric_type (str): Can be any of bleu, chrf, or ter depending on the metric to use.
            lang (str): Language code for the translation metric.
        """
        self.metric_type = metric_type
        self.lang = lang

    def get_metric(self):
        if self.metric_type == "bleu":
            import nltk

            nltk.download("punkt_tab")
            return sacrebleu.BLEU(trg_lang=self.lang)
        elif self.metric_type == "chrf":
            return sacrebleu.CHRF()
        elif self.metric_type == "chrf++":
            return sacrebleu.CHRF(word_order=2)
        elif self.metric_type == "ter":
            return sacrebleu.TER(asian_support=True if self.lang != "" else False)
        else:
            raise ValueError(f"Unknown corpus level translation metric type : {self.metric_type}")

    def compute_corpus(self, items: list[GenerativeCorpusMetricInput]) -> float:
        """Computes the metric score over all the corpus generated items, by using the sacrebleu implementation."""
        metric = self.get_metric()
        golds = [i.golds for i in items]
        preds = []
        for i in items:
            pred = as_list(i.preds)
            if len(pred) > 1:
                logger.info(
                    f"Multiple predictions present, keeping only the first prediction (when computing sacrebleu.{metric.__name__})."
                )
            preds.append(pred[0])

        if self.metric_type == "bleu":
            golds = [[gold[0] for gold in golds]]

        corpus_score = metric.corpus_score(hypotheses=preds, references=golds)
        score = corpus_score.score
        results = float(score)
        return results


class CorpusLevelPerplexityMetric(CorpusLevelComputation):
    def __init__(self, metric_type: str):
        """Stores the relevant parameter for a corpus level perplexity metric.
        Perplexity metrics compute more or less the same thing, which is a variation on the
        average of log-probabilities over a sequence, but the normalization and processing applied
        is different depending on the metric type.
        Perplexity uses an exponential and no weights for the average, weighted perplexity uses an exponential
        and the number of words as weights for the log-prob average, and bits per byte uses the number of bits
        for normalization and divides the results by log(2).

        Args:
            metric_type (str): Can be any of `perplexity`, `weighted_perplexity` or `bits_per_byte`
        """
        if metric_type not in ["perplexity", "weighted_perplexity", "bits_per_byte"]:
            raise ValueError(f"Unknown corpus level perplexity metric type : {metric_type}")

        self.metric_type = metric_type

    def compute_corpus(self, items: list[PerplexityCorpusMetricInput]):
        """Computes the metric score over all the corpus generated items."""
        logprobs = [i.logprobs for i in items]
        weights = [i.weights for i in items]

        if self.metric_type == "perplexity":
            return math.exp(-np.mean(logprobs))
        if self.metric_type == "weighted_perplexity":
            return math.exp(-sum(logprobs) / sum(weights))
        if self.metric_type == "bits_per_byte":
            return -sum(logprobs) / sum(weights) * 1 / math.log(2)


class MRewardBenchWeightedAccuracy(CorpusLevelComputation):
    """Computes weighted accuracy by category for M-RewardBench.

    This follows the m-rewardbench evaluation approach:
    1. Groups items by subset (source field)
    2. Computes accuracy per subset
    3. Weights subsets by example counts within each category
    4. Averages across categories with equal weights

    Reference: https://github.com/Cohere-Labs-Community/m-rewardbench
    """

    # Subset mapping from source to category
    SUBSET_MAPPING = {
        "Chat": [
            "alpacaeval-easy",
            "alpacaeval-length",
            "alpacaeval-hard",
            "mt-bench-easy",
            "mt-bench-med",
        ],
        "Chat Hard": [
            "mt-bench-hard",
            "llmbar-natural",
            "llmbar-adver-neighbor",
            "llmbar-adver-GPTInst",
            "llmbar-adver-GPTOut",
            "llmbar-adver-manual",
        ],
        "Safety": [
            "refusals-dangerous",
            "refusals-offensive",
            "xstest-should-refuse",
            "xstest-should-respond",
            "donotanswer",
        ],
        "Reasoning": [
            "math-prm",
            "hep-cpp",
            "hep-go",
            "hep-java",
            "hep-js",
            "hep-python",
            "hep-rust",
        ],
    }

    # Example counts per subset (from m-rewardbench)
    # Note: math-prm is upweighted to 983 (actual length 447) to match code subset count
    EXAMPLE_COUNTS = {
        "alpacaeval-easy": 79,
        "alpacaeval-length": 79,
        "alpacaeval-hard": 76,
        "mt-bench-easy": 24,
        "mt-bench-med": 38,
        "mt-bench-hard": 35,
        "math-prm": 983,
        "refusals-dangerous": 100,
        "refusals-offensive": 100,
        "llmbar-natural": 76,
        "llmbar-adver-neighbor": 124,
        "llmbar-adver-GPTInst": 87,
        "llmbar-adver-GPTOut": 42,
        "llmbar-adver-manual": 43,
        "xstest-should-refuse": 154,
        "xstest-should-respond": 247,
        "donotanswer": 135,
        "hep-cpp": 164,
        "hep-go": 164,
        "hep-java": 164,
        "hep-js": 164,
        "hep-python": 163,
        "hep-rust": 164,
    }

    def compute_corpus(self, items: list[LogprobCorpusMetricInput]) -> float:
        """Computes weighted accuracy by category for M-RewardBench.

        Args:
            items: List of LogprobCorpusMetricInput items

        Returns:
            float: Weighted accuracy score
        """
        subset_accuracies = {}
        subset_items: dict[str, list[tuple]] = {}

        for item in items:
            # Access the source from the item metadata if available
            subset = getattr(item, "source", "Unknown")

            if subset not in subset_items:
                subset_items[subset] = []
            subset_items[subset].append((item.golds, item.preds))

        # Compute accuracy per subset
        for subset, pairs in subset_items.items():
            correct = sum(1 for gold, pred in pairs if gold == pred)
            total = len(pairs)
            subset_accuracies[subset] = correct / total if total > 0 else 0.0

        # Compute weighted accuracy per category
        category_accuracies = {}
        for category, subsets in self.SUBSET_MAPPING.items():
            weighted_sum = 0.0
            total_examples = 0

            for subset in subsets:
                if subset in subset_accuracies:
                    count = self.EXAMPLE_COUNTS.get(subset, 0)
                    weighted_sum += subset_accuracies[subset] * count
                    total_examples += count

            category_accuracies[category] = (
                weighted_sum / total_examples if total_examples > 0 else 0.0
            )

        # Compute average across categories
        avg_accuracy = (
            sum(category_accuracies.values()) / len(category_accuracies) if category_accuracies else 0.0
        )

        # Log per-category scores and average
        logger.info("M-RewardBench Weighted Accuracy Results:")
        for category, acc in category_accuracies.items():
            logger.info(f"  {category}: {acc:.4f}")
        logger.info(f"  Average: {avg_accuracy:.4f}")

        return avg_accuracy
