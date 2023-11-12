# Inspired on https://github.com/mingu6/ProbFiltersVPR/blob/master/src/evaluation/evaluate.py
import numpy as np
from scipy.interpolate import interp1d
from sklearn import metrics


class PRCurve:
    def __init__(self, thres):
        self.thres = thres
        self.precisions = None
        self.recalls = None
        self.scores = None

    def generate_curve(self, dists, scores, interpolate=True):
        """
        Generates PR curves given true query and proposal poses.
        Can select interpolation of precision, where precision values
        are replaced with maximum precision for all recall values
        greater or equal to current recall.
        """
        scores_u = np.unique(scores)
        max_score = np.max(scores_u)
        self.scores = np.linspace(
            np.min(scores_u) - 1e-3, max_score + 1e-3, endpoint=True, num=1000
        )

        self.scores = np.flip(self.scores)
        self.precisions = np.ones_like(self.scores)
        self.recalls = np.zeros_like(self.scores)
        self.F1 = np.zeros_like(self.scores)
        for i, score_thres in enumerate(self.scores):
            localized = scores < score_thres
            
            correct = dists < self.thres
            # only count traverses with a proposal
            correct = np.logical_and(correct, localized)
            # compute precision and recall
            # index of -1 means not localized in max seq len
            nLocalized = np.count_nonzero(localized)
            nCorrect = np.count_nonzero(correct)
            if nLocalized > 0:
                # if none localized, precision = 1 by default
                self.precisions[i] = nCorrect / nLocalized
                if nCorrect + len(localized) - nLocalized > 0:
                    self.recalls[i] = nCorrect / (
                        nCorrect + len(localized) - nLocalized
                    )
        # flip curves for increasing recall
        self.precisions = np.flip(self.precisions)
        self.recalls = np.flip(self.recalls)
        self.scores = np.flip(self.scores)
        # ensure recalls are nondecreasing
        self.recalls, inds = np.unique(self.recalls, return_index=True)
        self.precisions = self.precisions[inds]
        self.scores = self.scores[inds]
        # chop off curve when recall first reaches 1
        ind_min = np.min(np.argwhere(self.recalls >= 1.0))
        self.recalls = np.sort(self.recalls[: ind_min + 1])
        self.precisions = np.sort(self.precisions[: ind_min + 1])[::-1]
        self.scores = np.sort(self.scores[: ind_min + 1])[::-1]
        if interpolate:
            for i in range(len(self.precisions)):
                self.precisions[i] = np.max(self.precisions[i:])
        
        return None

    def auc(self):
        if len(self.recalls) == 2: return 0.01
        elif len(self.recalls) < 2: return 0
        return metrics.auc(self.recalls, self.precisions)
    
    def clear(self):
        self.precisions = None
        self.recalls = None
        self.scores = None