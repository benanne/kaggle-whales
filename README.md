kaggle-whales
=============

Code for the Whale Detection Challenge competition on Kaggle. 

I used spherical k-means to learn features on whitened patches extracted convolutionally from contrast-normalised spectrograms, then max-pooled over those and trained SVMs / random forests / gradient boosting machines on that representation (random forests seemed to work best, in the end).

This was fairly fast so I could try out a lot of parameter settings. I performed a random search over some of the parameters (spectrogram size, patch size, normalisation, etc.) and averaged a few of the best models that came out of that.

I used the Enthought Python Distribution (EPD), numpy, scipy, matplotlib, sklearn.

This code hasn't been cleaned up, so it's quite messy in some places, largely undocumented, and some parts might be outdated (especially comments). I'm just uploading this in case anyone is interested in the details. My username on Kaggle is sedielem (http://www.kaggle.com/users/41181/sedielem).