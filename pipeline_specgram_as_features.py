"""
This script does everything from preprocessing to feature extraction to training to evaluation
"""

from matplotlib.mlab import specgram
import numpy as np
import scipy.signal as sig
from numpy.lib import stride_tricks
import time

import kmeans

from sklearn import cross_validation, ensemble, metrics, svm

import matplotlib.pyplot as plt 
plt.ion()
import utils

settings = {
    # prepro + specgram extraction
    'normalise_volume': True,
    'specgram_num_components': 128, # 64, 128, 256
    'specgram_redundancy': 4, # 1, 2, 4 -> 1 is no overlap between successive windows, 2 is half overlap, 4 is three quarter overlap
    'log_scale': 10**0,

    # zmuv
    'zmuv_bias': 0.0001,

    # classifier training /evaluation
    'n_folds': 4, # since we can do this in parallel, it's best to have a multiple of the number of cores
}

PLOT = True
DATA_PATH = "X_train.npy"
LABEL_PATH = "Y_train.npy"




def visualise_features(P, centroids):
    PC = np.dot(centroids, P.T)
    plt.figure(figsize=(12,8))
    utils.visualise_filters(PC.T, settings['patch_height'], settings['patch_width'], posneg=False)
    plt.draw()




start_time = time.time()
def tock():
    elapsed = time.time() - start_time
    print "  running for %.2f s" % elapsed


# load data
print "Load data"
X = np.load(DATA_PATH)
Y = np.load(LABEL_PATH)
tock()


# downsample
print "Downsample"
X_downsampled = sig.decimate(X, 2, axis=1)
# X_downsampled = X # DEBUG: NO DECIMATION
tock()


# normalise
if settings['normalise_volume']:
    print "Normalise volume"
    X_downsampled -= X_downsampled.mean(1).reshape(-1, 1)
    X_downsampled /= X_downsampled.std(1).reshape(-1, 1)
    tock()


# compute spectrograms
print "Compute spectrograms"
nfft = settings['specgram_num_components']
noverlap = nfft * (1 - 1. / settings['specgram_redundancy'])
log_scale = settings['log_scale']

dummy = specgram(X_downsampled[0], NFFT=nfft, noverlap=noverlap)[0] # to get the dimensions
X_specgram = np.zeros((X.shape[0], dummy.shape[0], dummy.shape[1]), dtype=X.dtype)

for k in xrange(X.shape[0]):
    X_specgram[k] = specgram(X_downsampled[k], NFFT=nfft, noverlap=noverlap)[0]

X_specgram = np.log(1 + log_scale * X_specgram)

tock()

features = X_specgram.reshape(X_specgram.shape[0], -1)

# zero mean unit variance
print "Zero mean unit variance on features"
zmuv_bias = settings['zmuv_bias']

fmeans = features.mean(0).reshape(1, -1)
fstds = features.std(0).reshape(1, -1)

features -= fmeans
features /= (fstds + zmuv_bias)

tock()

if PLOT:
    plt.figure()
    plt.imshow(features[:1000], cmap=plt.cm.binary, interpolation='none', vmin=-3, vmax=3)
    plt.title("some extracted features")


print "Do some memory cleanup"
del X
del X_downsampled
del X_specgram

# classifier training
print "Classifier training and evaluation through cross-validation"
n_folds = settings['n_folds']

# clf = ensemble.GradientBoostingClassifier()
# clf = ensemble.RandomForestClassifier(n_estimators=1024, min_samples_leaf=5, n_jobs=-1, verbose=1)
clf = svm.LinearSVC(C=10e-6)
scores = cross_validation.cross_val_score(clf, features, Y, cv=n_folds, score_func=metrics.auc_score, n_jobs=1) # n_jobs=-1) # use all cores

print "  auc: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std())

tock()
