import numpy as np
import samsatool
from samsatool import nearest_neighbour, kmeans
from samsatool import apply_loadings, pca_loadings

def generate_data(n=3,ndims=10,points_per_class=10):
  """
  Generate a synthetic dataset with n clusters for testing k-means.
  """
  X = []
  for i in range(n):
    X.append( i*3 + np.random.rand(points_per_class,ndims))
  return np.vstack(X)

def test_pca():
  """
  N.B. This test function simply runs code but does not check its output - this
       a "shallow" test.
  """
  X = generate_data(3,10,10)

  loadings, mean = pca_loadings( X, step=1, ndims=3 )
  assert loadings.shape == (10,3), "Error - loadings returned wrong shape (%s)." % loadings.shape

  pca = apply_loadings( X, loadings )
  assert pca.shape == (30,3), "Error - pca returned wrong shape (%s)." % pca.shape

def test_kmeans():
  X = generate_data(3,10,50)
  X=X.reshape( (15,10,10) ) # change shape of inputs to test shape independence
  loadings, mean = pca_loadings( X, step=2, ndims=2 ) # try to ensure test functions test a wide range of parameter configurations
  pca = apply_loadings( X, loadings )
  centroids, clss =  kmeans(pca, 3 )

  # check classification is correct
  clss = clss.ravel()
  assert len(np.unique(clss)) == 3, "Error - kmeans gave incorrect number of classes"
  assert (clss[:50] == clss[0]).all(), "Error - kmeans gave incorrect classification."
  assert (clss[50:100] == clss[50]).all(), "Error - kmeans gave incorrect classification."
  assert (clss[100:-1] == clss[100]).all(), "Error - kmeans gave incorrect classification."
  
