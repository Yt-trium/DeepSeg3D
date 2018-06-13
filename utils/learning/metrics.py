# ------------------------------------------------------------ #
#
# file : metrics.py
# author : CM
# Metrics for evaluation
#
# ------------------------------------------------------------ #
from keras import backend as K

# Sensitivity (true positive rate)
def sensitivity(truth, prediction):
    TP = K.sum(K.round(K.clip(truth * prediction, 0, 1)))
    P = K.sum(K.round(K.clip(truth, 0, 1)))
    return TP / (P + K.epsilon())

# Specificity (true negative rate)
def specificity(truth, prediction):
    TN = K.sum(K.round(K.clip((1-truth) * (1-prediction), 0, 1)))
    N = K.sum(K.round(K.clip(1-truth, 0, 1)))
    return TN / (N + K.epsilon())

# Precision (positive prediction value)
def precision(truth, prediction):
    TP = K.sum(K.round(K.clip(truth * prediction, 0, 1)))
    FP = K.sum(K.round(K.clip((1-truth) * prediction, 0, 1)))
    return TP / (TP + FP + K.epsilon())

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))
