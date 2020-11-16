# Adapted from score written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import numpy as np
np.seterr(divide='ignore', invalid='ignore')

class runningScore(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask],
            minlength=n_class ** 2,
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(
                lt.flatten(), lp.flatten(), self.n_classes
            )

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        # add evaluation matrix F1
        F1_score, mean_F1 = self.caluate_F1(hist)
        cls_F1 = dict(zip(range(self.n_classes), F1_score))
        fwIoU = self.fwIoU(hist,self.n_classes)
        acc = np.diag(hist).sum() / hist.sum()
        cls_acc = np.diag(hist) / hist.sum(axis=1)
        mean_acc = np.nanmean(cls_acc)
        cls_acc = dict(zip(range(self.n_classes), cls_acc))

        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))
        
        #5-class iou
        target = [x for x in range(self.n_classes)]
        hist_8 = hist[target]
        hist_8 = hist_8[:,target]

        iu_8 = np.diag(hist_8) / (hist_8.sum(axis=1) + hist_8.sum(axis=0) - np.diag(hist_8))
        mean_iu_8=np.nanmean(iu_8)




        return (
            {
                "Overall Acc : \t": acc,
                "Mean Acc : \t": mean_acc,
                "FreqW Acc : \t": fwavacc,
                "Mean IoU(9) : \t": fwIoU,
                "Mean IoU(8) : \t": mean_iu_8,
                "Mean F1 : \t": mean_F1,
                # "F1 score : \t": F1_score,
            },
            cls_iu, cls_acc, cls_F1
        )

    def caluate_F1(self, confusion_matrix):
        import pandas as pd
        F1_score = []
        for i in range(self.n_classes):
            p = confusion_matrix[i,i]/sum(confusion_matrix[:,i])
            R = confusion_matrix[i,i]/sum(confusion_matrix[i,:])
            F1 = 2/(1/p + 1/R)
            F1_score.append(F1)
        mean_F1 = np.asarray(F1_score).mean()

        return F1_score, mean_F1

    def fwIoU(self, confusion_matrix, num_classes):
        sum_sum_pij = 0
        fwIoU = 0
        for i in range(num_classes):# for every class
            IoU = 0
            pii = confusion_matrix[i][i]
            sum_pij = 0
            for j in range(num_classes):
                sum_pij += confusion_matrix[i][j]
            sum_pji = 0
            for j in range(num_classes):
                sum_pji += confusion_matrix[j][i]
            IoU = pii / (sum_pij + sum_pji - pii)
            fwIoU += IoU * sum_pij
            sum_sum_pij += sum_pij
        fwIoU = fwIoU / sum_sum_pij
        return fwIoU

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class averageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

