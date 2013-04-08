"""
This script averages predictions.

Usage: averager.py <predictions_file1> [predictions_file2] [...] <averaged_predictions_file>
"""

import numpy as np
import sys


if len(sys.argv) < 3:
    sys.exit("Usage: averager.py <predictions_file1> [predictions_file2] [...] <averaged_predictions_file>")

num_input_files = len(sys.argv) - 2

src_paths = []
for k in range(num_input_files):
    src_paths.append(sys.argv[k + 1])

tgt_path = sys.argv[-1]


print "Loading predictions"
scores_list = []
for src_path in src_paths:
    print "  %s" % src_path

    with open(src_path, 'r') as f:
        lines = f.readlines()

    scores = np.array([float(l.strip()) for l in lines])

    scores_list.append(scores)

print "Computing mean scores"
mean_scores = np.vstack(scores_list).mean(0)

# import pdb; pdb.set_trace()

print "Store averaged predictions"
with open(tgt_path, 'w') as f:
    for score in mean_scores:
        f.write(str(score) + '\n')

print "  stored in %s" % tgt_path
