import os

import numpy as np

from meta_define.attack import Strategic


class Report:

    def __init__(self):
        self.log = {"byzantine_records": []}

    def save(self, dir, name):
        os.makedirs(dir, exist_ok=True)
        np.save(dir + "/" + name, self.log)

    def global_record(self, context: dict):
        self.log.update(context)

    def byzantine_record(self, context: dict):
        updates = context["updates"]
        tag = context["tag"]
        strategic = context["strategic"]
        metric = {}
        TP, FP, TN, FN = count_Attack_metric(updates, tag, strategic)
        metric["TP"] = TP
        metric["FP"] = FP
        metric["TN"] = TN
        metric["FN"] = FN
        accuracy = (TP + TN) / (TP + FN + FP + TN) * 100
        precision = 0
        if TP + FP != 0:
            precision = TP / (TP + FP) * 100
        asr = 100 - precision
        recall = 0
        if TP + FN != 0:
            recall = TP / (TP + FN) * 100
        metric["accuracy"] = accuracy
        metric["precision"] = precision
        metric["recall"] = recall
        metric["asr"] = asr
        self.log["byzantine_records"].append(metric)
        print('accuracy:%.03f%% precision:%.03f%% recall:%.03f%% asr:%.03f%% ' % (accuracy, precision, recall, asr))


def count_metric(pred: list, label: list):
    TP, FP, TN, FN = 0, 0, 0, 0
    for i in range(len(pred)):
        if pred[i] == label[i] and label[i] == 1:
            TP += 1
        elif pred[i] == label[i] and label[i] == 0:
            TN += 1
        elif pred[i] != label[i] and label[i] == 1:
            FP += 1
        elif pred[i] != label[i] and label[i] == 0:
            FN += 1
    return TP, FP, TN, FN


def count_Attack_metric(gradients, tag: list, strategic: Strategic):
    result = strategic.examine({'gradients': gradients})
    return count_metric(result, tag)