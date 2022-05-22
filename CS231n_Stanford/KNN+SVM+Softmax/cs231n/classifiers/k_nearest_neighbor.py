import numpy as np



class KNearestNeighbor(object):
  """ a kNN classifier with L2 distance """

  def __init__(self):
    pass

  def train(self, X, y):
    """
    Train the classifier. For k-nearest neighbors this is just 
    memorizing the training data.

    Inputs:
    - X: A numpy array of shape (num_train, D) containing the training data
      consisting of num_train samples each of dimension D.
    - y: A numpy array of shape (N,) containing the training labels, where
         y[i] is the label for X[i].
    """
    self.X_train = X
    self.y_train = y
    
  def predict(self, X, k=1, num_loops=0):
    """
    Predict labels for test data using this classifier.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data consisting
         of num_test samples each of dimension D.
    - k: The number of nearest neighbors that vote for the predicted labels.
    - num_loops: Determines which implementation to use to compute distances
      between training points and testing points.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    if num_loops == 0:
      dists = self.compute_distances_no_loops(X)
    elif num_loops == 1:
      dists = self.compute_distances_one_loop(X)
    elif num_loops == 2:
      dists = self.compute_distances_two_loops(X)
    else:
      raise ValueError('Invalid value %d for num_loops' % num_loops)

    return self.predict_labels(dists, k=k)

  def compute_distances_two_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a nested loop over both the training data and the 
    test data.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data.

    Returns:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      is the Euclidean distance between the ith test point and the jth training
      point.
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))


    for i in range(num_test):
      for j in range(num_train):
        #####################################################################
        # TODO:                                                             #
        # Compute the l2 distance between the ith test point and the jth    #
        # training point, and store the result in dists[i, j]. You should   #
        # not use a loop over dimension.                                    #
        #####################################################################

        # L2 distance with numpy
        # dists[i,j] = np.sqrt(np.sum((X[i,:]-self.X_train[j,:])**2))

        # 문제 1: 위 구문(line: 78)을 numpy lib를 사용하지 않고 numpy lib를 사용한 결과와 동일하게 동작하도록 작성

        sub = []
        for test, train in zip(X[i,:], self.X_train[j,:]):
          sub.append( (test - train) ** 2 )

        sum = 0
        for s in sub:
          sum += s

        dists[i, j] = sum ** (1 / 2)
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
    return dists

  def compute_distances_one_loop(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a single loop over the test data.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in range(num_test):
      #######################################################################
      # TODO:                                                               #
      # Compute the l2 distance between the ith test point and all training #
      # points, and store the result in dists[i, :].                        #
      #######################################################################

      # L2 distance with numpy
      # dists[i,:] = np.sqrt(np.sum((X[i,:] - self.X_train)**2, axis = 1))

      # 문제 2: 위 구문(line: 105)을 numpy lib를 사용하지 않고 numpy lib를 사용한 결과와 동일하게 동작하도록 작성
      # (list to np array, tuple to np array 변환 함수(np.array())는 사용 가능)

      sub = []
      for row in self.X_train:
        sub.append((X[i,:] - row) ** 2)

      aux = []
      for array in sub:
        sum = 0
        for s in array:
          sum += s
        aux.append(sum ** (1 / 2))

      dists[i, :] = np.array(aux)

      #######################################################################
      #                         END OF YOUR CODE                            #
      #######################################################################
    return dists

  def compute_distances_no_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using no explicit loops.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train)) 
    #########################################################################
    # TODO:                                                                 #
    # Compute the l2 distance between all test points and all training      #
    # points without using any explicit loops, and store the result in      #
    # dists.                                                                #
    #                                                                       #
    # You should implement this function using only basic array operations; #
    # in particular you should not use functions from scipy.                #
    #                                                                       #
    # HINT: Try to formulate the l2 distance using matrix multiplication    #
    #       and two broadcast sums.                                         #
    #########################################################################

    # L2 distance vectorized with numpy
    # X_squared = np.sum(X**2,axis=1)
    # Y_squared = np.sum(self.X_train**2,axis=1)
    # XY = np.dot(X, self.X_train.T)
    # dists = np.sqrt(X_squared[:,np.newaxis] + Y_squared -2*XY)
    
    # 문제 3: 위 구문(line: 139~142)을 numpy lib를 사용하지 않고 numpy lib를 사용한 결과와 동일하게 동작하도록 작성
    # (list to np array, tuple to np array 변환 함수(np.array())는 사용 가능)

    aux = []
    for row in X ** 2:
      sum = 0
      for s in row:
        sum += s
      aux.append(sum)
    X_squared = np.array(aux)

    bux = []
    for row in self.X_train ** 2:
      sum = 0
      for s in row:
        sum += s
      bux.append(sum)
    Y_squared = np.array(bux)

    XY = []
    for _ in range(X.shape[0]):
      XY.append(list(range(self.X_train.shape[0])))

    for j in range(self.X_train.shape[0]):
      for i in range(X.shape[0]):
        sum = 0
        for k in range(X.shape[1]):
          sum += X[i][k] * self.X_train.T[k][j]
        XY[i][j] = sum

    XY = np.array(XY)

    reshape_X = []
    for _ in range(Y_squared.shape[0]):
      reshape_X.append(list(X_squared))
    X_s = np.array(reshape_X)

    reshape_Y = []
    for _ in range(X_squared.shape[0]):
      reshape_Y.append(list(Y_squared))
    Y_s = np.array(reshape_Y)

    dists = (X_s.T + Y_s -2*XY) ** (1/2)


    #########################################################################
    #                         END OF YOUR CODE                              #
    #########################################################################
    return dists

  def predict_labels(self, dists, k=1):
    """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.

    Inputs:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      gives the distance betwen the ith test point and the jth training point.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in range(num_test):
      # A list of length k storing the labels of the k nearest neighbors to
      # the ith test point.
      closest_y = []
      #########################################################################
      # TODO:                                                                 #
      # Use the distance matrix to find the k nearest neighbors of the ith    #
      # testing point, and use self.y_train to find the labels of these       #
      # neighbors. Store these labels in closest_y.                           #
      # Hint: Look up the function numpy.argsort.                             #
      #########################################################################

      test_row = dists[i,:]

      # sorted_row = np.argsort(test_row)

      # 문제 4-1: 위 구문(line: 181)을 numpy lib를 사용하지 않고 numpy lib를 사용한 결과와 동일하게 동작하도록 작성
      t_r = list(test_row)
      idx = list(range(len(t_r)))

      for j in range(len(t_r)):
        for v in range(j+1, len(t_r)):
          if t_r[j] > t_r[v]:
            t_r[j], t_r[v] = t_r[v], t_r[j]
            idx[j], idx[v] = idx[v], idx[j]

      sorted_row = idx

      closest_y = self.y_train[sorted_row[0:k]]

      #########################################################################
      # TODO:                                                                 #
      # Now that you have found the labels of the k nearest neighbors, you    #
      # need to find the most common label in the list closest_y of labels.   #
      # Store this label in y_pred[i]. Break ties by choosing the smaller     #
      # label.                                                                #
      #########################################################################

      # y_pred[i] = np.argmax(np.bincount(closest_y))
        
      # 문제 4-2: 위 구문(line:193)을 numpy lib를 사용하지 않고 numpy lib를 사용한 결과와 동일하게 동작하도록 작성
      # (list to np array, tuple to np array 변환 함수(np.array())는 사용 가능)
      for l in range(len(closest_y)):
        for m in range(l + 1, len(closest_y)):
          if closest_y[l] > closest_y[m]:
            closest_y[l], closest_y[m] = closest_y[m], closest_y[l]
      count = [0] * (closest_y[-1] + 1)

      for f in closest_y:
        count[f] += 1

      max = float("-inf")
      max_idx = 0
      for idx, value in enumerate(count):
        if value > max:
          max = value
          max_idx = idx

      y_pred[i] = max_idx
      #########################################################################
      #                           END OF YOUR CODE                            # 
      #########################################################################

    return y_pred

