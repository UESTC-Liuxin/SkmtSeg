# Adapted from score written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import numpy as np


class runningScore(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        #初始化混淆矩阵
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        """
        计算当前测试图相对的混淆矩阵
        :param label_true:
        :param label_pred:
        :param n_class:
        :return:
        """
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask],
            minlength=n_class ** 2,
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        """

        :param label_trues:
        :param label_preds:
        :return:
        """
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

        #计算PA
        acc = np.diag(hist).sum() / hist.sum()
        #计算单个类别的acc
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)

        #计算单个类别的iou
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        cls_iu = dict(zip(range(self.n_classes), iu))
        #mIoU
        mean_iu = np.nanmean(iu)

        #计算FwIou
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

        # 10-class iou
        target = [x for x in range(1,self.n_classes)]
        hist_10 = hist[target]
        hist_10 = hist_10[:, target]
        iu_10 = np.diag(hist_10) / (hist_10.sum(axis=1) + hist_10.sum(axis=0) - np.diag(hist_10))
        mean_iu_10 = np.nanmean(iu_10)

        return (
            {
                "Overall Acc : \t": acc,
                "Mean Acc : \t": acc_cls,
                "FreqW Acc : \t": fwavacc,
                "Mean IoU(11) : \t": mean_iu,
                "Mean IoU(10) : \t": mean_iu_10,
                "Mean F1 : \t": mean_F1,
            },
            cls_iu, cls_F1
        )

    def caluate_F1(self, confusion_matrix):
        F1_score = []
        for i in range(self.n_classes - 1):
            p = confusion_matrix[i, i] / sum(confusion_matrix[:, i])
            R = confusion_matrix[i, i] / sum(confusion_matrix[i, :])
            F1 = 2 / (1 / p + 1 / R)
            F1_score.append(F1)
        mean_F1 = np.asarray(F1_score).mean()
        # print('every class F1_score: {}. '.format(F1_score))
        # print('mean F1: {}. '.format(mean_F1))
        return F1_score, mean_F1

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
