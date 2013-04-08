"""
this glue script finds the N best models so far, generates predictions with them (if this has not happened yet) and averages them.
"""

import glob
import os
import cPickle as pickle
import numpy as np

import sys
import os


BASE_PATH = "/home/sander/projects/whales"

BEST_RESULTS_PATH = os.path.join(BASE_PATH, "best_results.pkl")
DATA_PATH = os.path.join(BASE_PATH, "X_train.npy")
LABEL_PATH = os.path.join(BASE_PATH, "Y_train.npy")
TEST_DATA_PATH = os.path.join(BASE_PATH, "X_test.npy")

RESULTS_DIR = os.path.join(BASE_PATH, "results")
PREDICTIONS_DIR = os.path.join(BASE_PATH, "predictions")

TARGET_PATH = os.path.join(PREDICTIONS_DIR, "predictions-final-best10.txt")


with open(BEST_RESULTS_PATH, 'r') as f:
    best_results_filenames = pickle.load(f)
best_results_filenames = best_results_filenames[-10:] # only 10


paths = [os.path.join(RESULTS_DIR, path) for path in best_results_filenames]

print "PATHS"
for i, path in enumerate(paths):
    print "%d: %s" % (i, path)
print


print "Generating predictions"
pred_paths = []
for i, path in enumerate(paths):
    pred_basename = os.path.basename(path).replace("results-", "predictions-").replace(".pkl", ".txt")
    pred_path = os.path.join("predictions", pred_basename)
    pred_paths.append(pred_path)
    # if os.path.exists(pred_path):
    #     print "Predictions file %s already exists, not regenerating" % pred_path
    #     continue

    # command = "epython generate_predictions.py %s %s" % (path, pred_path)
    # print command
    # os.system(command)

    if not os.path.exists(pred_path):
        sys.exit("ERROR: missing file %d: %s" % (i, pred_path))

print
print "Averaging predictions"
pred_paths_str = " ".join(pred_paths)
command = "epython averager.py %s %s" % (pred_paths_str, TARGET_PATH)
os.system(command)