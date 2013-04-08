"""
this glue script finds the N best models so far, generates predictions with them (if this has not happened yet) and averages them.
"""

import glob
import os
import cPickle as pickle
import numpy as np

import sys
import os

if len(sys.argv) != 3:
    sys.exit("Usage: best_n_ensemble.py <number_of_models> <target_path>")

N = int(sys.argv[1]) # number of models to average
tgt_path = sys.argv[2]


paths = glob.glob("results/results-*.pkl")

data = []

for path in paths:
    dirname = os.path.dirname(path)
    basename = os.path.basename(path)
    with open(path, 'r') as f:
        d = pickle.load(f)

    del d['params'] # save memory!
    d['path'] = path

    data.append(d)
    
all_aucs = [r['evaluation']['auc'] for r in data]

indices = np.argsort(all_aucs)
nbest_indices = indices[-N:]

paths = [data[k]['path'] for k in nbest_indices]
aucs = [data[k]['evaluation']['auc'] for k in nbest_indices]

print "Paths for n best results files so far:"
for path, auc in zip(paths, aucs):
    print "AUC %.4f  %s" % (auc, path)
print


print "Generating predictions"
pred_paths = []
for path in paths:
    pred_basename = os.path.basename(path).replace("results-", "predictions-").replace(".pkl", ".txt")
    pred_path = os.path.join("predictions", pred_basename)
    pred_paths.append(pred_path)
    if os.path.exists(pred_path):
        print "Predictions file %s already exists, not regenerating" % pred_path
        continue

    command = "epython generate_predictions.py %s %s" % (path, pred_path)
    print command
    os.system(command)

print
print "Averaging predictions"
pred_paths_str = " ".join(pred_paths)
command = "epython averager.py %s %s" % (pred_paths_str, tgt_path)
os.system(command)