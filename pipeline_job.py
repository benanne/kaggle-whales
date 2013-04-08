"""
This script does everything from preprocessing to feature extraction to training to evaluation with randomly sampled settings,
and then stores results.

2ND GEN: optional local normalisation, only 100 means. Gonna run a SHITTON of these.
"""

from matplotlib.mlab import specgram
import numpy as np
import scipy.signal as sig
from numpy.lib import stride_tricks
import time
import cPickle as pickle
from scipy import ndimage

import kmeans

from sklearn import cross_validation, ensemble, metrics, svm

import random_search as rs

num_components = 128 # rs.pick_one(64, 128)()
# patch_height = int(np.round(rs.uniform(0.2, 1.0)() * (num_components/2 + 1)))
patch_height = int(np.round(rs.uniform(0.5, 1.0)() * (num_components/2 + 1)))

settings = {
    # prepro + specgram extraction
    'normalise_volume': True,
    'specgram_num_components': num_components,
    'specgram_redundancy': rs.pick_one(2, 4)(), # 1 is no overlap between successive windows, 2 is half overlap, 4 is three quarter overlap
    'log_scale': rs.log_uniform(10**0, 10**1)(), # rs.log_uniform(10**0, 10**2)(), # rs.log_uniform(10**-1, 10**6)(),

    # local specgram normalisation
    'lnorm': True, # rs.pick_one(True, False)(),
    'lnorm_sigma_mean': rs.log_uniform(2, 8)(),
    'lnorm_sigma_std': rs.log_uniform(3, 7)(), # rs.log_uniform(2, 8)(),

    # patch extraction
    'patch_height': patch_height, # number of spectrogram components
    'patch_width': rs.uniform_int(6, 16)(), # number of timesteps
    'num_patches_for_learning': 100000,

    # whitening
    'retain': 1 - rs.log_uniform(10**-1, 10**-5)(),
    'pca_bias': 0.0001,

    # kmeans
    'num_means': 200,

    # extraction
    'threshold': None, # rs.pick_one(None, rs.uniform(0.0, 3.0)())(),

    # classifier training /evaluation
    'n_folds': 4, # since we can do this in parallel, it's best to have a multiple of the number of cores
}

TARGET_PATTERN = "/mnt/storage/usr/sedielem/whales/results/results-gen2-%s.pkl"
DATA_PATH = "/mnt/storage/usr/sedielem/whales/X_train.npy"
LABEL_PATH = "/mnt/storage/usr/sedielem/whales/Y_train.npy"

# TARGET_PATTERN = "results/results-gen2-%s.pkl"
# DATA_PATH = "X_train.npy"
# LABEL_PATH = "Y_train.npy"


expid = rs.generate_expid()
print "EXPERIMENT: %s" % expid
print
print settings
print


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

# import pdb; pdb.set_trace()

if settings['lnorm']:
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
print "Select subset of patches"
w, h = settings['patch_width'], settings['patch_height']
shape = X_specgram.shape
strides = X_specgram.strides
new_shape = (shape[0], shape[1] - h + 1, h, shape[2] - w + 1, w)
new_strides = (strides[0], strides[1], strides[1], strides[2], strides[2])
patches = stride_tricks.as_strided(X_specgram, shape=new_shape, strides=new_strides)

# now generate indices to select random patches
num = settings['num_patches_for_learning']
idx0 = np.random.randint(0, patches.shape[0], num)
idx1 = np.random.randint(0, patches.shape[1], num)
idx3 = np.random.randint(0, patches.shape[3], num)

patches_subset = patches[idx0, idx1, :, idx3, :].reshape(num, -1)

tock()


# whitening
print "Learn whitening"
retain = settings['retain']
bias = settings['pca_bias']

print "  computing transform..."
means = patches_subset.mean(0)
patches_subset_centered = patches_subset - means.reshape(1, -1)
cov = np.dot(patches_subset_centered.T, patches_subset_centered) / patches_subset.shape[0]
eigs, eigv = np.linalg.eigh(cov) # docs say the eigenvalues are NOT NECESSARILY ORDERED, but this seems to be the case in practice...

print "  computing number of components to retain %.2f of the variance..." % retain
normed_eigs = eigs[::-1] / np.sum(eigs) # maximal value first
eigs_sum = np.cumsum(normed_eigs)
num_components = np.argmax(eigs_sum > retain) # argmax selects the first index where eigs_sum > retain is true
print "  number of components to retain: %d" % num_components

P = eigv.astype('float32') * np.sqrt(1.0/(eigs + bias)) # PCA whitening
P = P[:, -num_components:] # truncate transformation matrix

tock()


# kmeans
print "Learn kmeans"
k = settings['num_means']

print "  whiten patches"
patches_subset_whitened = np.dot(patches_subset_centered, P)
print "  run kmeans"
centroids = kmeans.spherical_kmeans(patches_subset_whitened, k, num_iterations=20, batch_size=10000)

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
fmaxs = features.max(0)
fmins = features.min(0)

features -= fmins.reshape(1, -1)
features /= (fmaxs - fmins).reshape(1, -1)

tock()



print "Do some memory cleanup"
del X
del X_downsampled
del X_specgram
del patches
del patches_subset
del patches_subset_whitened


# classifier training
print "Classifier training and evaluation through cross-validation"
n_folds = settings['n_folds']

# clf = ensemble.GradientBoostingClassifier()
# clf = ensemble.RandomForestClassifier(n_estimators=1024, min_samples_leaf=5, n_jobs=-1, verbose=1)
clf = svm.LinearSVC(C=10e-3)
# clf = svm.SVC(kernel="linear", probability=True, C=10e-6)
# scores = cross_validation.cross_val_score(clf, features, Y, cv=n_folds, score_func=metrics.auc_score, n_jobs=1) # n_jobs=-1) # use all cores
# print "  auc: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std())

cv = cross_validation.StratifiedKFold(Y, n_folds)

def auc(t, y):
    fpr, tpr, thresholds = metrics.roc_curve(t, y)
    return metrics.auc(fpr, tpr)

predictions = np.zeros(Y.shape)
fold_aucs = []

for i, (train, test) in enumerate(cv):
    scores = clf.fit(features[train], Y[train]).decision_function(features[test])
    predictions[test] = scores

    fold_aucs.append(auc(Y[test], scores))

global_auc = auc(Y, predictions)


print "  auc over folds: %0.4f (+/- %0.4f)" % (np.mean(fold_aucs), np.std(fold_aucs))
print "  global auc: %0.4f" % global_auc

tock()


print "Storing experiment data"

evaluation = { 'auc': global_auc, 'fold_aucs': fold_aucs }
params = { 'means': means, 'P': P, 'centroids': centroids, 'fmaxs': fmaxs, 'fmins': fmins }

data = { 'params': params, 'settings': settings, 'evaluation': evaluation }

target_path = TARGET_PATTERN % expid

with open(target_path, 'w') as f:
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

print "  stored in %s" % target_path