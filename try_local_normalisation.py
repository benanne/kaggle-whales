
from matplotlib.mlab import specgram
import numpy as np
import scipy.signal as sig
from numpy.lib import stride_tricks
import time

from scipy.stats import norm
from scipy import ndimage

import matplotlib.pyplot as plt 
plt.ion()
import utils

import cPickle as pickle

# best settings so far
settings = {'lnorm': True,
  'lnorm_sigma_mean': 4.7004767598252784,
  'lnorm_sigma_std': 4.6898786142869611,
  'log_scale': 2.1850488896458171,
  'n_folds': 4,
  'normalise_volume': True,
  'num_means': 200,
  'num_patches_for_learning': 100000,
  'patch_height': 40,
  'patch_width': 7,
  'pca_bias': 0.0001,
  'retain': 0.98465982533888152,
  'specgram_num_components': 128,
  'specgram_redundancy': 2,
  'threshold': None}


DATA_PATH = "X_train.npy"
LABEL_PATH = "Y_train.npy"



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
X_downsampled = sig.decimate(X, 2, axis=1).astype(X.dtype)
# X_downsampled = X # DEBUG: NO DECIMATION
tock()

del X

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
X_specgram = np.zeros((X_downsampled.shape[0], dummy.shape[0], dummy.shape[1]), dtype=X_downsampled.dtype)

for k in xrange(X_downsampled.shape[0]):
    X_specgram[k] = specgram(X_downsampled[k], NFFT=nfft, noverlap=noverlap)[0]

X_specgram = np.log(1 + log_scale * X_specgram)

del X_downsampled

tock()

X_specgram_lnorm = np.zeros(X_specgram.shape, dtype=X_specgram.dtype)

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
        X_specgram_lnorm[k] = local_normalise(X_specgram[k], lnorm_sigma_mean, lnorm_sigma_std)

    tock()





print "Load info about classification difficulty"

with open("difficult_examples.pkl", 'r') as f:
    d = pickle.load(f)

X_pos = X_specgram[Y == 1]
X_pos_lnorm = X_specgram_lnorm[Y == 1]

gdn = X_pos[d['indices_pos'][4]]

idcs = d['indices_pos']

def rowcol_normalise(im, order='hv'):
    if order == 'hv':
        horiz = (im - im.mean(0)) / (im.std(0) + 0.001)
        vert = (horiz - horiz.mean(1).reshape(-1, 1)) / (horiz.std(1).reshape(-1, 1) + 0.001)
        return vert
    elif order == 'vh':
        vert = (im - im.mean(1).reshape(-1, 1)) / (im.std(1).reshape(-1, 1) + 0.001)
        horiz = (vert - vert.mean(0)) / (vert.std(0) + 0.001)
        return horiz
    elif order == 'sum':
        horiz = (im - im.mean(0)) / (im.std(0) + 0.001)
        vert1 = (horiz - horiz.mean(1).reshape(-1, 1)) / (horiz.std(1).reshape(-1, 1) + 0.001)
        vert = (im - im.mean(1).reshape(-1, 1)) / (im.std(1).reshape(-1, 1) + 0.001)
        horiz2 = (vert - vert.mean(0)) / (vert.std(0) + 0.001)
        return (vert1 + horiz2)


# def ta_normalise(im):
#     # remove temporal artifacts by dividing by the mean of the rows. vertical lines will then disappear.
#     return im - im.mean(0)

def ta_normalise(im, q=1.0, alpha=0.001):
    # remove temporal artifacts by dividing by the mean of the rows. vertical lines will then disappear.
    f1 = (im ** q).mean(0).reshape(1, -1) ** (1/q) + alpha
    return im / f1


def a_normalise(im, q=1.0, alpha=0.001):
    f1 = ((im ** q).mean(0).reshape(1, -1) ** (1/q)) + alpha
    f2 = ((im ** q).mean(1).reshape(-1, 1) ** (1/q)) + alpha
    return im / (f1 * f2)

def show_all(im, q=1.0, alpha=0.001):
    plt.figure(1)
    plt.imshow(im)
    plt.title("regular spectrogram")
    plt.draw()

    plt.figure(2)
    plt.imshow(local_normalise(im, lnorm_sigma_mean, lnorm_sigma_std))
    plt.title("locally-normalised spectrogram")
    plt.draw()

    plt.figure(3)
    plt.imshow(a_normalise(im, q, alpha))
    plt.title("a-normalised spectrogram")
    plt.draw()

    plt.figure(4)
    plt.imshow(local_normalise(a_normalise(im, q, alpha), lnorm_sigma_mean, lnorm_sigma_std))
    plt.title("a-normalised, then locally-normalised spectrogram")
    plt.draw()


# def show_all(im, smoothing=None):
#     plt.figure(1)
#     plt.imshow(im)
#     plt.title("regular spectrogram")
#     plt.draw()

#     plt.figure(2)
#     plt.imshow(local_normalise(im, lnorm_sigma_mean, lnorm_sigma_std))
#     plt.title("locally-normalised spectrogram")
#     plt.draw()

#     plt.figure(3)
#     plt.imshow(ta_normalise(im, smoothing))
#     plt.title("ta-normalised spectrogram")
#     plt.draw()

#     plt.figure(4)
#     plt.imshow(local_normalise(ta_normalise(im, smoothing), lnorm_sigma_mean, lnorm_sigma_std))
#     plt.title("ta-normalised, then locally-normalised spectrogram")
#     plt.draw()




# def show_both(im, order='hv', local_afterwards=False):
#     plt.figure(1)
#     plt.imshow(im)
#     plt.title("regular spectrogram")
#     plt.draw()
#     plt.figure(2)
#     if local_afterwards:
#         plt.imshow(local_normalise(rowcol_normalise(im, order=order), lnorm_sigma_mean, lnorm_sigma_std))
#     else:
#         plt.imshow(rowcol_normalise(im, order=order))
#     plt.title("rowcol-normalised spectrogram")
#     plt.draw()





# h = fspecial('gaussian', hsize, sigma) returns a rotationally symmetric Gaussian lowpass filter of size hsize with standard deviation sigma (positive). hsize can be a vector specifying the number of rows and columns in h, or it can be a scalar, in which case h is a square matrix. The default value for hsize is [3 3]; the default value for sigma is 0.5.


# function ln=localnormalize(IM,sigma1,sigma2)
# %LOCALNORMALIZE A local normalization algorithm that uniformizes the local
# %mean and variance of an image.
# %  ln=localnormalize(IM,sigma1,sigma2) outputs local normalization effect of 
# %  image IM using local mean and standard deviation estimated by Gaussian
# %  kernel with sigma1 and sigma2 respectively.
# %
# %  Contributed by Guanglei Xiong (xgl99@mails.tsinghua.edu.cn)
# %  at Tsinghua University, Beijing, China.
# epsilon=1e-1;
# halfsize1=ceil(-norminv(epsilon/2,0,sigma1));
# size1=2*halfsize1+1;
# halfsize2=ceil(-norminv(epsilon/2,0,sigma2));
# size2=2*halfsize2+1;
# gaussian1=fspecial('gaussian',size1,sigma1);
# gaussian2=fspecial('gaussian',size2,sigma2);
# num=IM-imfilter(IM,gaussian1);
# den=sqrt(imfilter(num.^2,gaussian2));
# ln=num./den;