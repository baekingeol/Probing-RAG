import re
import string
import collections
from typing import Tuple, List
import ftfy

from typing import Any, Dict


class Metric:
    """
    An abstract class representing a metric which can be accumulated.
    """

    def __call__(self, predictions: Any, gold_labels: Any):
        raise NotImplementedError

    def get_metric(self, reset: bool) -> Dict[str, Any]:
        """
        Compute and return the metric. Optionally also call `self.reset`.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """
        Reset any accumulators or internal state.
        """
        raise NotImplementedError

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


class EmF1Metric(Metric):
    def __init__(self) -> None:
        self._total_em = 0.0
        self._total_f1 = 0.0
        self._count = 0

    def __call__(
        self,
        predicted_answer: str,
        ground_truth_answers: List[str],
    ):
        #import pdb; pdb.set_trace()
        if isinstance(predicted_answer, list): predicted_answer = predicted_answer[0]
        if isinstance(ground_truth_answers[0], tuple): ground_truth_answers = [i for i in ground_truth_answers[0]]
        #import pdb; pdb.set_trace()
        predicted_answer = ftfy.fix_text(predicted_answer)
        ground_truth_answers = [ftfy.fix_text(e) for e in ground_truth_answers]
        
        assert isinstance(predicted_answer, str)
        assert isinstance(ground_truth_answers, (Tuple, List))

        exact_scores = metric_max_over_ground_truths(compute_exact, predicted_answer, ground_truth_answers)
        f1_scores = metric_max_over_ground_truths(compute_f1, predicted_answer, ground_truth_answers)

        self._total_em += int(exact_scores)
        self._total_f1 += f1_scores
        self._count += 1

    def get_metric(self, reset: bool = False) -> Tuple[float, float]:
        exact_match = self._total_em / self._count if self._count > 0 else 0
        f1_score = self._total_f1 / self._count if self._count > 0 else 0
        if reset:
            self.reset()
        return {"em": round(exact_match, 3), "f1": round(f1_score, 3), "count": self._count}

    def reset(self):
        self._total_em = 0.0
        self._total_f1 = 0.0
        self._count = 0
        
def compute_metrics(predicted_support: List[str], gold_support: List[str]) -> Dict:
    # Taken from hotpot_eval

    predicted_support = set([re.sub(r" +", "", ftfy.fix_text(str(e)).lower()) for e in predicted_support])
    gold_support = set([re.sub(r" +", "", ftfy.fix_text(str(e)).lower()) for e in gold_support])

    tp, fp, fn = 0, 0, 0
    for e in predicted_support:
        if e in gold_support:
            tp += 1
        else:
            fp += 1
    for e in gold_support:
        if e not in predicted_support:
            fn += 1
    prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    em = 1.0 if fp + fn == 0 else 0.0

    # In case everything is empty, set both f1, em to be 1.0.
    # Without this change, em gets 1 and f1 gets 0
    if not predicted_support and not gold_support:
        f1, em = 1.0, 1.0
        f1, em = 1.0, 1.0

    return {"prec": prec, "recall": recall, "f1": f1, "em": em}


class SupportEmF1Metric(Metric):
    """
    SupportMetric: Em and F1 (Similar to HotpotQA Sp metric)
    """

    def __init__(self, do_normalize_answer: bool = False) -> None:
        self._titles_total_em = 0.0
        self._titles_total_f1 = 0.0
        self._titles_total_precision = 0.0
        self._titles_total_recall = 0.0

        self._paras_total_em = 0.0
        self._paras_total_f1 = 0.0
        self._paras_total_precision = 0.0
        self._paras_total_recall = 0.0

        self._total_predicted_titles = 0
        self._max_predicted_titles = -float("inf")
        self._min_predicted_titles = float("inf")

        self._total_predicted_paras = 0
        self._max_predicted_paras = -float("inf")
        self._min_predicted_paras = float("inf")

        self._do_normalize_answer = do_normalize_answer
        self._count = 0

    def __call__(self, predicted_support: List[str], gold_support: List[str]):

        predicted_support = predicted_support or []

        if self._do_normalize_answer:
            predicted_support = [normalize_answer(e) for e in predicted_support]
            gold_support = [normalize_answer(e) for e in gold_support]

        if not gold_support:
            gold_support_titles = []
            gold_support_paras = []
            predicted_support_titles = predicted_support_paras = predicted_support

        elif gold_support[0].startswith("pid"):
            for e in gold_support + predicted_support:
                assert e.startswith("pid")
            predicted_support_titles = [e.split("___")[1] for e in predicted_support]
            predicted_support_paras = predicted_support
            gold_support_titles = [e.split("___")[1] for e in gold_support]
            gold_support_paras = gold_support

        else:
            for e in gold_support + predicted_support:
                assert not e.startswith("pid")
            predicted_support_titles = predicted_support_paras = predicted_support
            gold_support_titles = gold_support_paras = gold_support

        predicted_support_titles = set(map(str, predicted_support_titles))
        predicted_support_paras = set(map(str, predicted_support_paras))

        gold_support_titles = set(map(str, gold_support_titles))
        gold_support_paras = set(map(str, gold_support_paras))

        titles_metrics = compute_metrics(predicted_support_titles, gold_support_titles)
        paras_metrics = compute_metrics(predicted_support_paras, gold_support_paras)

        self._total_predicted_titles += len(predicted_support_titles)
        self._max_predicted_titles = max(self._max_predicted_titles, len(predicted_support_titles))
        self._min_predicted_titles = min(self._min_predicted_titles, len(predicted_support_titles))

        self._total_predicted_paras += len(predicted_support_paras)
        self._max_predicted_paras = max(self._max_predicted_titles, len(predicted_support_paras))
        self._min_predicted_paras = min(self._min_predicted_titles, len(predicted_support_paras))

        self._titles_total_em += float(titles_metrics["em"])
        self._titles_total_f1 += titles_metrics["f1"]
        self._titles_total_precision += titles_metrics["prec"]
        self._titles_total_recall += titles_metrics["recall"]

        self._paras_total_em += float(paras_metrics["em"])
        self._paras_total_f1 += paras_metrics["f1"]
        self._paras_total_precision += paras_metrics["prec"]
        self._paras_total_recall += paras_metrics["recall"]

        self._count += 1

    def get_metric(self, reset: bool = False) -> Tuple[float, float]:
        """
        Returns
        -------
        Average exact match and F1 score (in that order).
        """
        titles_exact_match = self._titles_total_em / self._count if self._count > 0 else 0
        titles_f1_score = self._titles_total_f1 / self._count if self._count > 0 else 0
        titles_precision_score = self._titles_total_precision / self._count if self._count > 0 else 0
        titles_recall_score = self._titles_total_recall / self._count if self._count > 0 else 0

        paras_exact_match = self._paras_total_em / self._count if self._count > 0 else 0
        paras_f1_score = self._paras_total_f1 / self._count if self._count > 0 else 0
        paras_precision_score = self._paras_total_precision / self._count if self._count > 0 else 0
        paras_recall_score = self._paras_total_recall / self._count if self._count > 0 else 0

        avg_predicted_titles = self._total_predicted_titles / self._count if self._count > 0 else 0
        avg_predicted_paras = self._total_predicted_paras / self._count if self._count > 0 else 0

        if reset:
            self.reset()

        return {
            "title_em": round(titles_exact_match, 3),
            "title_f1": round(titles_f1_score, 3),
            "title_precision": round(titles_precision_score, 3),
            "title_recall": round(titles_recall_score, 3),
            "para_em": round(paras_exact_match, 3),
            "para_f1": round(paras_f1_score, 3),
            "para_precision": round(paras_precision_score, 3),
            "para_recall": round(paras_recall_score, 3),
            "avg_predicted_titles": avg_predicted_titles,
            "max_predicted_titles": self._max_predicted_titles,
            "min_predicted_titles": self._min_predicted_titles,
            "avg_predicted_paras": avg_predicted_paras,
            "max_predicted_paras": self._max_predicted_paras,
            "min_predicted_paras": self._min_predicted_paras,
            "count": self._count,
        }

    def reset(self):

        self._titles_total_em = 0.0
        self._titles_total_f1 = 0.0
        self._titles_total_precision = 0.0
        self._titles_total_recall = 0.0

        self._paras_total_em = 0.0
        self._paras_total_f1 = 0.0
        self._paras_total_precision = 0.0
        self._paras_total_recall = 0.0

        self._total_predicted_titles = 0
        self._max_predicted_titles = -float("inf")
        self._min_predicted_titles = float("inf")

        self._total_predicted_paras = 0
        self._max_predicted_paras = -float("inf")
        self._min_predicted_paras = float("inf")

        self._count = 0