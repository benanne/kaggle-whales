"""
This script 'biases' predictions based on the order of the examples. Naughty!

Usage: smoother.py <predictions_file> <smoothed_predictions_file>
"""

import numpy as np
import scipy.signal as sig
import sys


LABELS_PATH = "Y_train.npy"

MA_FACTOR = 110.0 #  100.0 # 200.0 # amount of moving average smoothing
SCALE = 0.3 # 1.0 # amount of biasing (additive)


if len(sys.argv) != 3:
    sys.exit("Usage: smoother.py <predictions_file> <smoothed_predictions_file>")

src_path = sys.argv[1]
tgt_path = sys.argv[2]

print "Loading predictions"
with open(src_path, 'r') as f:
    lines = f.readlines()

scores = np.array([float(l.strip()) for l in lines])

print "Loading training labels"
Y = np.load(LABELS_PATH)

print "Smoothing and rescaling training labels"
Ys = np.convolve(Y, np.ones(MA_FACTOR) / MA_FACTOR, mode='same')
Ysr = sig.resample(Ys, scores.shape[0])

print "Biasing predictions"
scores_biased = scores + SCALE * (Ysr - 0.5)


print "Store biased predictions"
with open(tgt_path, 'w') as f:
    for score in scores_biased:
        f.write(str(score) + '\n')

print "  stored in %s" % tgt_path
