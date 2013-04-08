import numpy as np


# import theano
# import theano.tensor as T
# x, y = T.matrices('x', 'y')
# dot = theano.function([x,y], T.dot(x,y))

dot = np.dot


def spherical_kmeans(X, k, num_iterations=10, batch_size=10000, damped=True, one_hot=False):
    """
    k-means constrained to the unit sphere, for whitened input data, optionally with damped centroid updates.
    As described in Coates & Ng, 2012.

    Make sure X is whitened!

    X:              whitened data (num_datapoints, dim)

    k:              number of means

    num_iterations: self-explanatory

    batch_size:     self-explanatory

    damped:         use damping when updating the centroids, to avoid centroids with few assigned data points from jumping around.

    one_hot:        instead of computing a numerical value for the assignments, as Coates proposes,
                    just set it to 1 if the mean is selected, and 0 otherwise (one hot coding).
    """

    num_datapoints, dim = X.shape
    num_batches = int(np.ceil(num_datapoints / float(batch_size)))

    # initialise centroids: random on the unit sphere
    centroids = np.random.normal(0,1, (k, dim)).astype(X.dtype)
    centroids /= np.sqrt((centroids**2).sum(1)).reshape(-1, 1) # normalise to unit length

    shuffle_indices = np.arange(num_datapoints)

    for i in xrange(num_iterations):
        print "iteration %d" % i
        np.random.shuffle(shuffle_indices) # shuffle data for every iteration

        # compute similarities and assignments
        print "  compute similarities and assignments"
        assignments = np.zeros(num_datapoints, dtype='int32')
        coefficients = np.ones(num_datapoints, dtype=X.dtype)

        for j in xrange(num_batches):
            s = slice(j * batch_size, (j + 1) * batch_size)
            similarities = dot(X[s, :], centroids.T)
            assignments[s] = np.argmax(similarities, 1)
            if not one_hot:
                coefficients[s] = similarities[np.arange(similarities.shape[0]), assignments[s]]
                # if one_hot, the coefficients are always 1 and they are never updated

        counts = np.sum(np.atleast_2d(assignments) == np.atleast_2d(np.arange(k)).T, 1)
        # print np.max(counts), np.min(counts), np.sum(counts==0)

        # update centroids
        print "  update centroids"
        for ik in xrange(k):
            if counts[ik] == 0:
                print "WARNING: cluster is empty, resetting centroid."
                centroids[ik] = X[shuffle_indices[ik], :] # if a centroid is empty, reinitialise it with a random training example.
            else:
                new_centroid = np.sum(coefficients[assignments == ik].reshape(-1, 1) * X[assignments == ik, :], 0)
                if damped:
                    centroids[ik] += new_centroid
                else:
                    centroids[ik] = new_centroid

        centroids /= np.sqrt((centroids**2).sum(1)).reshape(-1, 1) # normalise to unit length

    return centroids



def encode_hard(X, centroids, one_hot=False):
    """
    Perform single centroid assignment encoding (the default 'encoder' for kmeans)
    """
    similarities = dot(X, centroids.T)
    assignments = np.argmax(similarities, 1)
    features = np.zeros((X.shape[0], centroids.shape[0]))
    if one_hot:
        coefficients = 1
    else:
        coefficients = similarities[np.arange(X.shape[0]), assignments]
    features[np.arange(X.shape[0]), assignments] = coefficients
    return features


def encode_triangle(X, centroids):
    """
    Perform triangle k-means encoding
    """
    X3 = X.reshape(X.shape[0], 1, X.shape[1])
    centroids3 = centroids.reshape(1, centroids.shape[0], centroids.shape[1])
    z = np.sqrt(((X3 - centroids3) ** 2).sum(2))
    means = z.mean(1).reshape(-1, 1)
    return np.maximum(means - z, 0)


def encode_threshold(X, centroids, threshold=0):
    """
    Perform encoding with a threshold function
    """
    return np.maximum(dot(X, centroids.T) - threshold, 0)


def encode_linear(X, centroids):
    """
    Linear encoding, just the dot product
    """
    return dot(X, centroids.T)


def encode_abs(X, centroids):
    """
    Features are absolute values of the linear encoding
    """
    return abs(dot(X, centroids.T))

