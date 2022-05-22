import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  num_classes = W.shape[1]
  num_train = X.shape[0]

  # scores = np.dot(X, W)
  # 문제 6-1: 위 구문(line:36)을 numpy lib를 사용하지 않고 numpy lib를 사용한 결과와 동일하게 동작하도록 작성

  scores = []

  for _ in range(X.shape[0]):
    scores.append(list(range(W.shape[1])))

  for j in range(W.shape[1]):
    for i in range(X.shape[0]):
      sum = 0
      for k in range(X.shape[1]):
        sum += X[i][k] * W[k][j]
      scores[i][j] = sum

  scores = np.array(scores)

  for ii in range(num_train):
    current_scores = scores[ii, :] #(1,10)

    shift_scores = current_scores - np.max(current_scores)

    # loss_ii = -shift_scores[y[ii]] + np.log(np.sum(np.exp(shift_scores)))
    # 문제 6-2: 위 구문(line:43)을 numpy lib를 사용하지 않고 numpy lib를 사용한 결과와 동일하게 동작하도록 작성
    import math
    sum = 0
    for s in shift_scores:
      sum += math.exp(s)
    loss_ii = math.log(sum) - shift_scores[y[ii]]

    loss += loss_ii

    for jj in range(num_classes):
      # softmax_score = np.exp(shift_scores[jj]) / np.sum(np.exp(shift_scores))
      # 문제 6-3: 위 구문(line:43)을 numpy lib를 사용하지 않고 numpy lib를 사용한 결과와 동일하게 동작하도록 작성
      import math
      exp_sum = 0
      for s in shift_scores:
        exp_sum += math.exp(s)
      softmax_score = math.exp(shift_scores[jj]) / exp_sum

      if jj == y[ii]:
        dW[:, jj] += (-1 + softmax_score) * X[ii]
      else:
        dW[:, jj] += softmax_score * X[ii]

  loss /= num_train
  # loss += reg * np.sum(W*W)
  # 문제 6-4: 위 구문(line:43)을 numpy lib를 사용하지 않고 numpy lib를 사용한 결과와 동일하게 동작하도록 작성
  sum = 0
  for row in range((W ** 2).shape[0]):
    for col in range((W ** 2).shape[1]):
      sum += (W ** 2)[row][col]
  loss += reg * sum

  dW /= num_train
  dW += 2*reg*W

  # (list to np array, tuple to np array 변환 함수(np.array())는 사용 가능)

    
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  num_train = X.shape[0]

  scores = np.dot(X, W)
  # shift_scores = scores - np.max(scores, axis=1)[...,np.newaxis]
  # 문제 7-1: 위 구문(line: 96)을 numpy lib를 사용하지 않고 numpy lib를 사용한 결과와 동일하게 동작하도록 작성

  max_list = []
  for row in scores:
    max = float("-inf")
    for s in row:
      if s > max:
        max = s
    max_list.append(max)

  reshape_max = []
  for _ in range(scores.shape[1]):
    reshape_max.append(list(max_list))
  max_matrix = np.array(reshape_max)

  shift_scores = scores - max_matrix.T




  # softmax_scores = np.exp(shift_scores)/ np.sum(np.exp(shift_scores), axis=1)[..., np.newaxis]
  # 문제 7-2: 위 구문(line: 99)을 numpy lib를 사용하지 않고 numpy lib를 사용한 결과와 동일하게 동작하도록 작성
  import math
  r = []
  m = []
  for row in shift_scores:
    sum = 0
    for s in row:
      sum += math.exp(s)

    r = []
    for k in row:
      r.append(math.exp(k) / sum)

    m.append(r)
  softmax_scores = np.array(m)




  dScore = softmax_scores
  dScore[range(num_train),y] = dScore[range(num_train),y] - 1

  dW = np.dot(X.T, dScore)
  dW /= num_train
  dW += 2*reg*W

  correct_class_scores = np.choose(y, shift_scores.T)  # Size N vector

  # loss = -correct_class_scores + np.log(np.sum(np.exp(shift_scores), axis=1))
  # loss = np.sum(loss)
  # 문제 7-3: 위 구문(line: 110, 111)을 numpy lib를 사용하지 않고 numpy lib를 사용한 결과와 동일하게 동작하도록 작성
  log_sum = []
  loss = []

  for row in shift_scores:
    sum = 0
    for s in row:
      sum += math.exp(s)
    log_sum.append(math.log(sum))

  for idx in range(correct_class_scores.shape[0]):
    loss.append(log_sum[idx] - correct_class_scores[idx])

  sum_l = 0
  for l in loss:
    sum_l += l

  loss = sum_l



  loss /= num_train
  # loss += reg * np.sum(W*W)
  # 문제 7-4: 위 구문(line: 110, 111)을 numpy lib를 사용하지 않고 numpy lib를 사용한 결과와 동일하게 동작하도록 작성

  sum = 0
  for i in range((W ** 2).shape[0]):
    for j in range((W ** 2).shape[1]):
      sum += (W ** 2)[i][j]

  loss += reg * sum

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

