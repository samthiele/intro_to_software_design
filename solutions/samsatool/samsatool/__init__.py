import numpy as np

def pca_loadings( data, step=1, ndims=3):
    """
    Compute PCA loadings for the specified dataset.
    
    Args:
     - data (np.ndarray): a numpy array for which the last dimensions contains feature 
                          values (bands) used to compute the PCA.
     - step (int): if greater than one, every nth data point will be used for estimating the 
                   covariance matrix. This can increase performance for big datasets. Default is 1.
     - ndims (int): the number of loading vectors to create. This should equal the number of 
                    dimensions to retain during PCA dimensionality reduction. Default is 3.
                    
    Returns:
     - loadings (np.ndarray): the estimated loadings vectors with shape (data.shape[-1],ndims)
     - mean (float): the mean centering that was used to derive these loadings (and should be used when applying
                         them).
    """
    
    # asserts can help identify problems early
    assert isinstance(data, np.ndarray), "Error: Data must be a numpy array" 
    
    data = data.reshape((-1, data.shape[-1])) # reshape array to a feature vector of shape (npixels,nbands)
    mean = np.mean(data, axis=0)
    X = data - mean # mean center the data
    X = X[::step] # only look at a subset of the data, useful for big images
    cov = np.dot(X.T, X) / (X.shape[0] - 1)  # compute covariance matrix
    
    eigval, eigvec = np.linalg.eig(cov) # get eigenvalues of covariance matrix
    idx = np.argsort(np.abs(eigval[:ndims]))[::-1] # sort them into descending order
    eigvec = eigvec[:, idx]
    eigval = np.abs(eigval[idx])
    
    return eigvec, mean 

def apply_loadings( data, loadings, mean=None ):
    """
    Apply loadings to a dataset to derive a dimensionality reduce set of features. Both arguments should be numpy
    arrays. Note that the provided data will be mean-centered first, either using the provided mean or the mean of
    the datasets. Returns a numpy array with the same dimensions as data, except for the last dimension which will
    have one element for every loading vector.
    
    ^^ Note that while this docstring contains the same information as the previous one,
       it is much harder to read!
    """
    # reshape array to a feature vector of shape (npixels,nbands)
    X = data.reshape((-1, data.shape[-1])) 
    
    # do mean centering
    if mean is None:
        mean = np.mean(X, axis=0)
    X = X - mean
    
    # apply loadings
    out = np.zeros(data.shape[:-1]+(loadings.shape[-1],)) # create output array of the correct shape
    for b in range(0, loadings.shape[-1]):
        out[..., b] = np.dot(data, loadings[:, b])
    return out

def nearest_neighbour(data, centroids):
    """
    Apply a nearest neighbour classifier.
    
    Args:
        data (np.ndarray): the data to classify. The last dimension of data must have the same length
                          as the last dimension of centroids.
        centroids (np.ndarray): an (n,ndims) array of n centroids.
     
    Returns:
        class : an array with the same shape as data (except the last dimension) containing class ids.
    """
    X = data.reshape((-1, data.shape[-1])) # flatten into feature vector. Notice that I've reused this code
                                              # three times now, so could consider pulling it into its own function!
                                              # (but, for now I won't)
    dist = np.linalg.norm( (X[None,:,:] - centroids[:,None,:]), axis=-1)
    clss = np.argmin( dist, axis=0 )
    return clss.reshape(data.shape[:-1])

def kmeans(data, n, seed=42, maxiter=100):
    """
    Find the kmeans centroids (and associated classificaiton) for a dataset
    
    Args:
        data (np.ndarray): the data array to find centroids for. The last dimension of this 
                          should contain the features.
        n (int): the number of classes to extract.
        seed (int): a random seed to use when initialising class centroids. Default is 42, naturally.
        maxiter (int): the maximum number of iterations before throwing an error. Default is 100.
     
    Returns:
        - centroids (np.ndarray): an (n,ndims) array of n centroids.
        - classification (np.ndarray): an array of class ids corresponding to the kmeans classified data.
    """
    
    # get random initial centroids
    np.random.seed(seed)
    X = data.reshape((-1, data.shape[-1])) # flatten into feature vector
    centroids = X[ np.random.choice(X.shape[0], n, replace=False), : ]
    
    # compute k means
    delta = np.array([np.inf, np.inf, np.inf])
    i = 0
    while np.max(delta > 0.0001) and i < maxiter:
        clss = nearest_neighbour( X, centroids )
        means = np.array([np.mean( X[clss==i, :], axis=0) for i in range(n)])
        delta = np.linalg.norm(centroids - means, axis=1 )
        centroids = means
        i += 1
    return centroids, clss.reshape(data.shape[:-1])
