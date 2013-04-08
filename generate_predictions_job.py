"""
generate final predictions job
"""

from matplotlib.mlab import specgram
import numpy as np
import scipy.signal as sig
from numpy.lib import stride_tricks
import time
import cPickle as pickle
from scipy import ndimage

import kmeans


import sys
import os

sys.path.insert(0, "/mnt/storage/usr/avdnoord/sklearn/")
from sklearn import cross_validation, ensemble, metrics, svm


BASE_PATH = "/mnt/storage/usr/sedielem/whales"
# BASE_PATH = "/home/sander/projects/whales"

DATA_PATH = os.path.join(BASE_PATH, "X_train.npy")
LABEL_PATH = os.path.join(BASE_PATH, "Y_train.npy")

TEST_DATA_PATH = os.path.join(BASE_PATH, "X_test.npy")

BEST_RESULTS_PATH = os.path.join(BASE_PATH, "best_results.pkl")

RESULTS_DIR = os.path.join(BASE_PATH, "results")
PREDICTIONS_DIR = os.path.join(BASE_PATH, "predictions")

if len(sys.argv) != 2:
    sys.exit("Usage: generate_predictions_job.py <index>")

index = int(sys.argv[1])

with open(BEST_RESULTS_PATH, 'r') as f:
    best_results_filenames = pickle.load(f)

results_path = os.path.join(RESULTS_DIR, best_results_filenames[index])
print "Using results in %s" % results_path


pred_basename = os.path.basename(results_path).replace("results-", "predictions-").replace(".pkl", ".txt")
predictions_path = os.path.join(PREDICTIONS_DIR, pred_basename)
print "Storing predictions in %s" % predictions_path


with open(results_path, 'r') as f:
    results = pickle.load(f)

settings = results['settings']
params = results['params']
P = params['P']
means = params['means']
centroids = params['centroids']
fmins = params['fmins']
fmaxs = params['fmaxs']


print "GENERATING PREDICTIONS FOR: %s" % results_path
print
print settings
print


start_time = time.time()
def tock():
    elapsed = time.time() - start_time
    print "  running for %.2f s" % elapsed


# load data
print "Load training data"
X = np.load(DATA_PATH)
Y = np.load(LABEL_PATH)
tock()

# downsample
print "Downsample"
X_downsampled = sig.decimate(X, 2, axis=1).astype(X.dtype)
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

if 'lnorm' in settings and settings['lnorm']: # check presence first for backwards compatibility
    print "Locally normalise spectrograms"
    lnorm_sigma_mean = settings['lnorm_sigma_mean']
    lnorm_sigma_std = settings['lnorm_sigma_std']

    def local_normalise(im, sigma_mean, sigma_std):
        """
        based on matlab code by Guanglei Xiong, see http://www.mathworks.com/matlabcentral/fileexchange/8303-local-normalization
        """
        means = ndimage.gaussian_filter(im, sigma_mean)
        im_centered = im - means
        stds = np.sqrt(ndimage.gaussian_filter(im_centered**2, sigma_std))
        return im_centered / stds

    for k in xrange(X_specgram.shape[0]):
        X_specgram[k] = local_normalise(X_specgram[k], lnorm_sigma_mean, lnorm_sigma_std)

    tock()


# patch extraction
print "Patch extraction"
w, h = settings['patch_width'], settings['patch_height']
shape = X_specgram.shape
strides = X_specgram.strides
new_shape = (shape[0], shape[1] - h + 1, h, shape[2] - w + 1, w)
new_strides = (strides[0], strides[1], strides[1], strides[2], strides[2])
patches = stride_tricks.as_strided(X_specgram, shape=new_shape, strides=new_strides)

tock()


# feature extraction and summarisation
print "Feature extraction and summarisation"
threshold = settings['threshold']
PC = np.dot(P, centroids.T)

# def summarise_features(features):
#     features = features.reshape(features.shape[0], -1, features.shape[3]) # merge time and frequency axes
#     return np.hstack([features.mean(1), features.std(1), features.min(1), features.max(1)]) # summarize over time axis
#     # return features.mean(1)
#     # return features.max(1)

def summarise_features(features):
    features_freq_pooled = features.max(1) # max pool over frequency
    n_timesteps = features_freq_pooled.shape[1]
    # quadrant pooling over time
    parts = [features_freq_pooled.max(1)] # max pooling over the whole timelength
    n_timeslices = 4
    slice_size = n_timesteps // n_timeslices # floor
    for k in xrange(n_timeslices):
        features_slice = features_freq_pooled[:, k*slice_size:(k+1)*slice_size].max(1)
        parts.append(features_slice)

    return np.hstack(parts)

    # return features_freq_pooled.max(1) # time pooling



def extract_features(batch, means, PC, threshold=None):
    batch_shape = batch.shape
    batch = batch.transpose(0,1,3,2,4).reshape(-1, h * w)

    features = np.dot(batch - means.reshape(1, -1), PC)

    if threshold is not None: # if no threshold specified, use linear features
        features = np.maximum(features - threshold, 0) # thresholding

    # features = features.reshape(batch_shape[0], -1, PC.shape[1]) # split examples and other axes

    features = features.reshape(batch_shape[0], batch_shape[1], batch_shape[3], PC.shape[1]) # (examples, frequency bins, time, features)

    # import pdb; pdb.set_trace()

    features = summarise_features(features)
    return features

batch_size = 100
num_examples = patches.shape[0]
num_batches = int(np.ceil(num_examples / float(batch_size)))
features = np.zeros((num_examples, 5 * PC.shape[1]), dtype=X.dtype)
for b in xrange(num_batches):
    print "  batch %d of %d" % (b+1, num_batches)
    batch = patches[b*batch_size:(b+1)*batch_size]
    current_batch_size = batch.shape[0]
    batch_features = extract_features(batch, means, PC, threshold)
    features[b*batch_size:b*batch_size + current_batch_size] = batch_features

tock()


# interval normalisation
print "Normalisation of features"
features -= fmins.reshape(1, -1)
features /= (fmaxs - fmins).reshape(1, -1)

tock()

print "Do some memory cleanup"
del X
del X_downsampled
del X_specgram
del patches


# classifier training
print "Classifier training"

# clf = svm.LinearSVC(C=10e-3)
# clf = ensemble.RandomForestClassifier(n_estimators=200, n_jobs=4, verbose=1)
clf = ensemble.GradientBoostingClassifier(n_estimators=50, min_samples_split=2, max_features=int(np.sqrt(1000)))
clf.fit(features, Y)

tock()


print "Further cleanup: no longer need training data"
del features
del Y


# load test data
print "Load test data"
X_test = np.load(TEST_DATA_PATH).astype('float32')

tock()

# downsample
print "Downsample (test)"
X_downsampled = sig.decimate(X_test, 2, axis=1).astype(X_test.dtype)
tock()

# normalise
if settings['normalise_volume']:
    print "Normalise volume"
    X_downsampled -= X_downsampled.mean(1).reshape(-1, 1)
    X_downsampled /= X_downsampled.std(1).reshape(-1, 1)
    tock()


# compute spectrograms
print "Compute spectrograms (test)"
nfft = settings['specgram_num_components']
noverlap = nfft * (1 - 1. / settings['specgram_redundancy'])
log_scale = settings['log_scale']

dummy = specgram(X_downsampled[0], NFFT=nfft, noverlap=noverlap)[0] # to get the dimensions
X_specgram = np.zeros((X_test.shape[0], dummy.shape[0], dummy.shape[1]), dtype=X_test.dtype)

for k in xrange(X_test.shape[0]):
    X_specgram[k] = specgram(X_downsampled[k], NFFT=nfft, noverlap=noverlap)[0]

X_specgram = np.log(1 + log_scale * X_specgram)

del X_test
del X_downsampled

tock()

if 'lnorm' in settings and settings['lnorm']:
    print "Locally normalise spectrograms"
    lnorm_sigma_mean = settings['lnorm_sigma_mean']
    lnorm_sigma_std = settings['lnorm_sigma_std']

    for k in xrange(X_specgram.shape[0]):
        X_specgram[k] = local_normalise(X_specgram[k], lnorm_sigma_mean, lnorm_sigma_std)

    tock()

# patch extraction
print "Patch extraction (test)"
w, h = settings['patch_width'], settings['patch_height']
shape = X_specgram.shape
strides = X_specgram.strides
new_shape = (shape[0], shape[1] - h + 1, h, shape[2] - w + 1, w)
new_strides = (strides[0], strides[1], strides[1], strides[2], strides[2])
patches = stride_tricks.as_strided(X_specgram, shape=new_shape, strides=new_strides)

tock()


# feature extraction and summarisation
print "Feature extraction and summarisation (test)"
threshold = settings['threshold']
PC = np.dot(P, centroids.T)

batch_size = 100
num_examples = patches.shape[0]
num_batches = int(np.ceil(num_examples / float(batch_size)))
features = np.zeros((num_examples, 5 * PC.shape[1]), dtype=X_specgram.dtype)
for b in xrange(num_batches):
    print "  batch %d of %d" % (b+1, num_batches)
    batch = patches[b*batch_size:(b+1)*batch_size]
    current_batch_size = batch.shape[0]
    batch_features = extract_features(batch, means, PC, threshold)
    features[b*batch_size:b*batch_size + current_batch_size] = batch_features

tock()


# interval normalisation
print "Normalisation of features (test)"
features -= fmins.reshape(1, -1)
features /= (fmaxs - fmins).reshape(1, -1)

tock()

print "Do some memory cleanup (again)"
del X_specgram
del patches

print "Compute predictions"
# scores = clf.decision_function(features)
scores = clf.predict_proba(features)[:, 1]



print "Store predictions"
with open(predictions_path, 'w') as f:
    for score in scores:
        f.write(str(float(score)) + '\n')

print "  stored in %s" % predictions_path







