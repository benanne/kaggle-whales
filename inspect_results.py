"""
load results files and classification results files into an array of dicts
"""

import glob
import os
import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
plt.ion()


paths = glob.glob("results/results-*.pkl")
# paths = glob.glob("results/results-gen2-*.pkl")

data = []

DEL_PARAMS = True # save space by not keeping all the parameters in memory

for path in paths:
    dirname = os.path.dirname(path)
    basename = os.path.basename(path)
    with open(path, 'r') as f:
        d = pickle.load(f)

    d['path'] = path

    if DEL_PARAMS:
        del d['params']
    
    data.append(d)
    
    
    
    
def point_plot(result_sets, xt, yt, *args, **kwargs):
    """
    make a point plot based on the result set, where xt is the value type used 
    for the x axis (a result set dictionary key), and yt is the same thing for
    the y axis.
    """
    xs = [r[xt] for r in result_sets]
    ys = [r[yt] for r in result_sets]
    plt.scatter(xs, ys, *args, **kwargs)
    
    
def pp(result_sets, xt, yt, logx=False, logy=False):
    """
    make a point plot based on the result set, where xt is the value type used 
    for the x axis (a result set dictionary key), and yt is the same thing for
    the y axis.
    """
    if logx:
        xs = [np.log(r[xt]) for r in result_sets]
    else:
        xs = [r[xt] for r in result_sets]
    if logy:
        ys = [np.log(r[yt]) for r in result_sets]
    else:
        ys = [r[yt] for r in result_sets]
    plt.scatter(xs, ys)
    
    
def point_plot_lambda(result_sets, fx, fy, *args, **kwargs):
    """
    like point_plot, but with lambdas that operate on the result_set to select
    the values to plot.
    """
    xs = [fx(r) for r in result_sets]
    ys = [fy(r) for r in result_sets]
    plt.scatter(xs, ys, *args, **kwargs)
    
    
def nbest(result_sets, n=10, key='valid'):
    rs = list(result_sets) # make a copy because we'll sort in place
    rs.sort(key=lambda d: d[key])
    return rs[-n:]
    
    

aucs = np.array([r['evaluation']['auc'] for r in data])
order = np.argsort(aucs)



# filenames = [os.path.basename(data[k]['path']) for k in order[-30:]]
