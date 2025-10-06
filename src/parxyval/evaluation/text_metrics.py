from statistics import mean
from typing import Callable, Dict

from nltk import edit_distance as nltk_edit_distance
from nltk import f_measure
from nltk import precision as nltk_precision
from nltk import recall as nltk_recall
from nltk.translate import meteor_score as nltk_meteor_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from parxy_core.models import Document

from parxyval.evaluation.factory import register_metric
from parxyval.evaluation.utils import text_similarity


def pagewise_metric(
    doc1: Document,
    doc2: Document,
    score_fn: Callable[[str, str], float],
    tokenize: bool = True,
) -> float:
    """Generic page-wise metric aggregator.

    Args:
        doc1 : Document
        doc2 : Document
        score_fn : Callable[[list[str], list[str]], float] or Callable[[str, str], float]
            Function that computes a similarity score for a pair of pages.
        tokenize : bool
            If True, texts are split into tokens before passing to score_fn.
            If False, raw page.text strings are passed.

    Returns
    -------
    float
        Average score across pages.
    """
    scores = []
    for page1, page2 in zip(doc1.pages, doc2.pages):
        x, y = (
            (page1.text.split(), page2.text.split())
            if tokenize
            else (page1.text, page2.text)
        )
        score = score_fn(x, y)
        scores.append(score if score is not None else 0.0)

    return mean(scores) if scores else 0.0


@register_metric('sequence_matcher')
def sequence_matcher_metric(reference: Document, test: Document) -> Dict[str, float]:
    return {
        'sequence_matcher': pagewise_metric(
            reference, test, lambda a, b: text_similarity(a=a, b=b), tokenize=False
        )
    }


@register_metric('jaccard_similarity')
def jaccard_similarity_metric(reference: Document, test: Document) -> Dict[str, float]:
    return {
        'jaccard_similarity': pagewise_metric(
            reference,
            test,
            lambda x, y: len(set(x) & set(y)) / len(set(x) | set(y)) if x or y else 0,
            tokenize=True,
        )
    }


@register_metric('bleu_score')
def bleu_score(reference: Document, test: Document) -> Dict[str, float]:
    smooth = SmoothingFunction().method1
    return {
        'bleu_score': pagewise_metric(
            reference,
            test,
            lambda x, y: sentence_bleu(
                references=[x], hypothesis=y, smoothing_function=smooth
            ),
            tokenize=True,
        )
    }


@register_metric('f1_score')
def f1_score(reference: Document, test: Document) -> Dict[str, float]:
    return {
        'f1_score': pagewise_metric(
            reference, test, lambda x, y: f_measure(set(x), set(y)) or 0, tokenize=True
        )
    }


@register_metric('precision')
def precision(reference: Document, test: Document) -> Dict[str, float]:
    return {
        'precision': pagewise_metric(
            reference,
            test,
            lambda x, y: nltk_precision(set(x), set(y)) or 0,
            tokenize=True,
        )
    }


@register_metric('recall')
def recall(reference: Document, test: Document) -> Dict[str, float]:
    return {
        'recall': pagewise_metric(
            reference,
            test,
            lambda x, y: nltk_recall(set(x), set(y)) or 0,
            tokenize=True,
        )
    }


@register_metric('edit_distance')
def edit_distance(reference: Document, test: Document) -> Dict[str, float]:
    return {
        'edit_distance': pagewise_metric(
            reference,
            test,
            lambda x, y: nltk_edit_distance(x, y) / max(len(x), len(y))
            if not (len(x) == len(y) == 0)
            else 0,
            tokenize=True,
        )
    }


@register_metric('meteor_score')
def meteor_score(reference: Document, test: Document) -> Dict[str, float]:
    return {
        'meteor': pagewise_metric(
            reference,
            test,
            lambda x, y: nltk_meteor_score.meteor_score([x], y) or 0,
            tokenize=True,
        )
    }
