# ------------------------------------------------------------ #
#
# file : tools/evaluator.py
# author : CM
# Evaluate matching between an image and its ground truth
#
# ------------------------------------------------------------ #
import os
import sys
import nibabel as nib
import numpy as np
from matplotlib import pyplot
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, \
    jaccard_similarity_score, f1_score

# Prediction
filename_pr = sys.argv[1]
if(not os.path.isfile(filename_pr)):
    sys.exit(1)
# Input
filename_in = sys.argv[2]
if(not os.path.isfile(filename_in)):
    sys.exit(1)
# Ground truth
filename_gd = sys.argv[3]
if(not os.path.isfile(filename_gd)):
    sys.exit(1)

data_pr = nib.load(filename_pr).get_data()
data_in = nib.load(filename_in).get_data()
data_gd = nib.load(filename_gd).get_data()

data_pr = data_pr.flatten()
data_in = data_in.flatten()
data_gd = data_gd.flatten()

data_in = data_in/data_in.max()

""""
print(data_pr.min())
print(data_pr.max())
print(data_in.min())
print(data_in.max())
print(data_gd.min())
print(data_gd.max())
"""

# Sørensen–Dice coefficient
dice1 = np.sum(data_pr[data_gd==1])*2.0 / (np.sum(data_pr) + np.sum(data_gd))
dice2 = np.sum(data_in[data_gd==1])*2.0 / (np.sum(data_in) + np.sum(data_gd))

print("Sørensen–Dice coefficient with Prediction", dice1)
print("Sørensen–Dice coefficient with Raw Input", dice2)


# Average precision score
print("Average precision score with Prediction", average_precision_score(data_gd, data_pr))
print("Average precision score with Raw Input", average_precision_score(data_gd, data_in))


# F1 binary score
f1_score_pr = []
f1_score_in = []
f1_threshold = []

for i in range(0, 11):
    threshold = i*0.1
    data_pr_threshold = data_pr > threshold
    data_in_threshold = data_in > threshold

    f1_threshold.append(threshold)
    f1_score_pr.append(f1_score(data_gd, data_pr_threshold))
    f1_score_in.append(f1_score(data_gd, data_in_threshold))

f1_curve = pyplot.figure()

pyplot.plot(f1_threshold, f1_score_pr, '-', label='PR')
pyplot.plot(f1_threshold, f1_score_in, '-', label='IN')

pyplot.title('F1 curve')
pyplot.xlabel("threshold")
pyplot.ylabel("F1 score")
pyplot.legend(loc="lower right")
pyplot.savefig("F1.png")


# Jaccard similarity score

j_score_pr = []
j_score_in = []
j_threshold = []

for i in range(0, 11):
    threshold = i*0.1
    data_pr_threshold = data_pr > threshold
    data_in_threshold = data_in > threshold

    j_threshold.append(threshold)
    j_score_pr.append(jaccard_similarity_score(data_gd, data_pr_threshold))
    j_score_in.append(jaccard_similarity_score(data_gd, data_in_threshold))

j_curve = pyplot.figure()

pyplot.plot(j_threshold, j_score_pr, '-', label='PR')
pyplot.plot(j_threshold, j_score_in, '-', label='IN')

pyplot.title('Jaccard curve')
pyplot.xlabel("threshold")
pyplot.ylabel("Jaccard score")
pyplot.legend(loc="lower right")
pyplot.savefig("Jaccard.png")


# Receiver Operating Characteristic
fpr1, tpr1, thresholds1 = roc_curve(data_gd, data_pr)
fpr2, tpr2, thresholds2 = roc_curve(data_gd, data_in)

AUC_ROC1 = roc_auc_score(data_gd, data_pr)
AUC_ROC2 = roc_auc_score(data_gd, data_in)

roc_curve = pyplot.figure()
pyplot.plot(fpr1, tpr1, '-', label='PR Area Under the Curve (AUC = %0.4f)' % AUC_ROC1)
pyplot.plot(fpr2, tpr2, '-', label='IN Area Under the Curve (AUC = %0.4f)' % AUC_ROC2)
pyplot.title('ROC curve')
pyplot.xlabel("FPR (False Positive Rate)")
pyplot.ylabel("TPR (True Positive Rate)")
pyplot.legend(loc="lower right")
pyplot.savefig("ROC.png")


# Precision-Recall
precision1, recall1, thresholds1 = precision_recall_curve(data_gd, data_pr)
precision2, recall2, thresholds2 = precision_recall_curve(data_gd, data_in)

pr_curve = pyplot.figure()

pyplot.plot(recall1, precision1, '-', label='PR')
pyplot.plot(recall2, precision2, '-', label='IN')

pyplot.title('PR curve')
pyplot.xlabel("Recall")
pyplot.ylabel("Precision")
pyplot.legend(loc="lower right")
pyplot.savefig("PR.png")
