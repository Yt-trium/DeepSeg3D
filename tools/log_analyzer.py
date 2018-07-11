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

with open(input_filename, 'r') as csvfile:
    log = csv.DictReader(csvfile, delimiter=',')
    for row in log:
        epoch.append(int(row["epoch"]))

        train_loss.append(float(row["loss"]))
        valid_loss.append(float(row["val_loss"]))

        train_precision.append(float(row["precision"]))
        valid_precision.append(float(row["val_precision"]))

fig = pyplot.figure()

axs0 = fig.add_subplot(2, 1, 1)

axs0.plot(epoch, train_loss, label='train loss')
axs0.plot(epoch, valid_loss, label='valid loss')
axs0.set_ylabel('loss')
pyplot.legend()

axs1 = fig.add_subplot(2, 1, 2)

axs1.plot(epoch, train_precision, label='train precision')
axs1.plot(epoch, valid_precision, label='valid precision')
axs1.set_ylabel('precision')
pyplot.legend()

pyplot.xlabel('epoch')
pyplot.show()