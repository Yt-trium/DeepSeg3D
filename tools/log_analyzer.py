# ------------------------------------------------------------ #
#
# file : tools/log_analyzer.py
# author : CM
# Tools for "training.log" analysis
#
# ------------------------------------------------------------ #
import os
import sys
import csv
import numpy as np
from matplotlib import pyplot

input_filename  = sys.argv[1]

if not os.path.isfile(input_filename):
    sys.exit("invalid argument")

epoch = []
train_loss = []
valid_loss = []
train_precision = []
valid_precision = []
train_sensitivity = []
valid_sensitivity = []
train_specificity = []
valid_specificity = []

with open(input_filename, 'r') as csvfile:
    log = csv.DictReader(csvfile, delimiter=',')
    for row in log:
        epoch.append(int(row["epoch"]))

        train_loss.append(float(row["loss"]))
        valid_loss.append(float(row["val_loss"]))

        train_precision.append(float(row["precision"]))
        valid_precision.append(float(row["val_precision"]))

        train_sensitivity.append(float(row["sensitivity"]))
        valid_sensitivity.append(float(row["val_sensitivity"]))

        train_specificity.append(float(row["specificity"]))
        valid_specificity.append(float(row["val_specificity"]))

fig = pyplot.figure()

axs0 = fig.add_subplot(3, 1, 1)

axs0.plot(epoch, train_loss, label='train loss')
axs0.plot(epoch, valid_loss, label='valid loss')
axs0.set_ylabel('loss')
pyplot.legend()

axs1 = fig.add_subplot(3, 1, 3)

axs1.plot(epoch, train_precision, label='train precision')
axs1.plot(epoch, valid_precision, label='valid precision')
axs1.set_ylabel('precision')
pyplot.legend()

pyplot.xlabel('epoch')

axs2 = fig.add_subplot(3, 2, 3)

axs2.plot(epoch, train_sensitivity, label='train sensitivity')
axs2.plot(epoch, valid_sensitivity, label='valid sensitivity')
axs2.set_ylabel('sensitivity')

axs3 = fig.add_subplot(3, 2, 4)

axs3.plot(epoch, train_specificity, label='train specificity')
axs3.plot(epoch, valid_specificity, label='valid specificity')
axs3.set_ylabel('specificity')

pyplot.show()