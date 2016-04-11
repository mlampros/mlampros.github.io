---
layout: post
title: Linear and logistic regression in Theano 
tags: [Theano, python]
comments: true
---


This blog post shows how to use the theano library to perform linear and logistic regression. I won't go into details of what linear or logistic regression is, because the purpose of this post is mainly to use the theano library in regression tasks. However, details on [linear](https://en.wikipedia.org/wiki/Linear_regression) and [logistic regression](https://en.wikipedia.org/wiki/Logistic_regression) can be found on the Wikipedia website. For the purpose of this blog post, I created a small python package **Regression_theano**, which resides in my Github repository. So, assuming that you,<br>

* do want to install it, and you work on a Linux PC (note that it is possible to continue reading the post without installing the package)
* have already [installed and configured theano](http://deeplearning.net/software/theano/install.html#install)

Then you,

* can download the zip file from the [repository](https://github.com/mlampros/Regression_theano)
* unzip it
* open the unziped file (it includes a *setup.py* file) 
* run **python setup.py install** to install it.

By the way, a tutorial on how to create a package from python code can be found [here](http://python-packaging.readthedocs.org/en/latest/index.html).
<br>
<br>

#### Theano 

A few words about theano : Theano is a Python library with a NumPy-like syntax, that lets you define, optimize, and evaluate mathematical expressions. It can be compiled and run efficiently on either CPU or GPU and it can be as fast as the C language on a CPU and orders of magnitude faster when using a GPU. *Tutorials* on how to start using the theano library can be found [here](http://deeplearning.net/software/theano/tutorial/index.html#tutorial) and *machine learning implementations* like a logistic regression, a Multilayer perceptron or a deep convolutional neural network, [here](http://www.deeplearning.net/tutorial/).
<br>
<br>

#### Regression in theano

When implementing a simple linear or logistic regression model in theano the first thing to do is to declare the variables and functions. Theano then creates a [symbolic graph](http://deeplearning.net/software/theano/extending/graphstructures.html#graphstructures), which we can use with our inputs to obtain results. In the *Regression* python Class of the *Regression_theano* package, first, I define X and y, 

*Linear regression*:

* X : tensor Matrix
* y : tensor **vector**

*Logistic regression*:

* X : tensor Matrix
* y : tensor **Matrix**   (in classification tasks the response variable should be one-hot-encoded)


Then, I initialize the weights of the parameters ( here I use a glorot uniform initialization as explained in this [jupyter notebook](http://nbviewer.jupyter.org/github/vkaynig/ComputeFest2015_DeepLearning/blob/master/Fully_Connected_Networks.ipynb)),
```py

init_weights_params = theano.shared(np.asarray(np.random.uniform(low = -np.sqrt(6. / (inp + outp)), high = np.sqrt(6. / (inp + outp)), size = shape), 
                                                                 dtype=theano.config.floatX)) # shape == dims of the weights matrix

```

and the weights for the bias ( which default to zero ),

```py

init_weights_bias = theano.shared(np.asarray(0, dtype=theano.config.floatX))                                      # for linear regression is a single value

init_weights_bias = theano.shared(np.zeros((self.y.shape[1],), dtype=theano.config.floatX))                       # for logistic regression is an array of values

```

Both the *init_weights_params* and the *init_weights_bias* are theano-shared-variables, meaning they are shared between different functions and different function calls. In the module, there is also the option to exclude the bias from the calculations.

In the same way, the predictions can be obtained using,

```py

py_x = T.dot(X, init_weights_params) + init_weights_bias                             # In case of linear regression 

py_x = T.nnet.softmax(T.dot(X, init_weights_params) + init_weights_bias)             # In case of logistic regression 

```

The aim of training a linear or logistic model in theano is to minimize/maximize the objective function until a global minimum/maximum is reached. In case of the linear model the objective can be for instance the mean-squared-error or the root-mean-squared-error, whereas in logistic regression the objective can be the binary-crossentropy or the categorical crossentropy ( a user-defined-objective is also possible),

```py

cost = T.sqr(py_x - Y)/float(nrows_X)                    # linear regression

cost = T.nnet.binary_crossentropy(py_x, Y)               # logistic regression
```

Regularization is a method to prevent overfitting and can be added to a model in 3 ways : L1 , L2 or L1 + L2 (elastic-net),

```py

# L1, L2 can be between 0.0 and 1.0 ( 0.0 is no regularization )

reg_param_L1  = abs(T.sum(init_weights_params) + T.sum(init_weights_bias))                               # L1 regrularization

reg_param_L2 = T.sum(T.sqr(init_weights_params)) + T.sum(T.sqr(init_weights_bias))                       # L2 regularization

cost = cost + L1 * reg_param_L1 + L2 * reg_param_L2

```

The next step is to create a theano.function for compilation. This function will take the index of all the train data as *input* and deploy an optimizer (such as sgd, rmsprop, adagrad) to update the parameters at each epoch returning an *output* (here the cost),

```py

train = theano.function(inputs = [index], 
                        
                        outputs = cost,  
                        
                        updates = sgd(cost, Params, learning_rate),
                        
                        givens = { X: train_X[0:index], Y: train_y[0:index] }, 
                        
                        allow_input_downcast = True)

```

A nice thing in theano is that the gradients are computed automatically using the function *theano.gradient.grad(cost = cost, wrt = Params)*.

Before proceeding a rule of thumb here is to use the *givens* argument in combination with an *index* inside the theano.function as long as the data fits in the GPU memory (if GPU is used). That way the data will reside in the GPU and we can spare unnecessary transfers of data from the CPU to GPU ([discussion on this issue](https://groups.google.com/forum/#!topic/theano-users/GwcSuaRUbG8)).

Considering the previous *train* function, the *givens* argument takes as inputs two shared variables, which should be declared at the beginning together with X and y,

```py

index = T.lscalar('index')                          # tensor variable index should equal the number of rows of the train data

train_X = shared_dataset(X_train_data)              # shared variable X_train for the givens argument

train_y = self.shared_dataset(y_train_data)         # shared variable y_train for the givens argument

test_X = shared_dataset(X_test)                     # shared variable, will be used for evaluation        

```

If on the other hand a batch-size is employed due to the data size, then in addition to the previous variables, batch indices should be built, 

```py

batch_size = 128                   # example of batch_size

n_train_batches = train_X.get_value(borrow=True).shape[0] / batch_size

n_test_batches = test_X.get_value(borrow=True).shape[0] / batch_size

```

and then the *train* function will look as follows,

```py

train = theano.function(inputs = [index], 
                        
                        outputs = cost, 
                        
                        updates = sgd(cost, Params, learning_rate),
                        
                        givens = { X: train_X[index * batch_size: (index + 1) * batch_size],
                                              
                                   Y: train_y[index * batch_size: (index + 1) * batch_size]}, 
                        
                        allow_input_downcast = True)
```

The only thing that is different in this function is the *givens* argument, where *train_X*, *train_y* is split into batches.

Another important thing is when to stop train a model, especially in the case of iterative models (which is the case here). [Early stopping](http://deeplearning.net/tutorial/code/DBN.py) is implemented in the deeplearning tutorials, however, I found this [monitor function](https://henri.io/posts/using-gradient-boosting-with-early-stopping.html), which is meant for gradient boosting, quite useful and I've adjusted my Regression class accordingly. So, learning of the model stops when the calculated loss is increased/decreased for a consecutive number of epochs,

```py

predict_valid = theano.function(inputs = [index],                                 
                                
                                outputs = py_x, 
                                
                                givens = { X: test_X[0:index]},     # returns the predictions when all train data are used
                                
                                allow_input_downcast = True)



predict_valid = theano.function(inputs = [index],                                  
                                
                                outputs = py_x,                               
                                
                                # returns the predictions when batch-index of train data is used
                                
                                givens = { X: test_X[index * batch_size: (index + 1) * batch_size]},        
                                
                                allow_input_downcast = True)

```

A final step is to use a separate prediction function to predict unknown data once training ends,

```py

predict = theano.function(inputs = [X], 
                          
                          outputs = py_x)

```

This function will take as input the new data set and it will return the predictions. 
<br><br>
The explained code-chunks can be found in the [Regression_theano.py](https://github.com/mlampros/Regression_theano/blob/master/Regression_theano/Regression_theano.py) file.

<br>

#### Example data sets

I'll test the *Theano_regression* package using the [Boston data](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html) for regression and the [Mnist data](https://www.kaggle.com/c/digit-recognizer/data) for classification,

```py

from Regression_theano import Regression
from sklearn import preprocessing
from sklearn.datasets import load_boston
from sklearn import metrics
import numpy as np
import random


# REGRESSION

boston = load_boston()
print(boston.data.shape)

X = boston['data']
X = preprocessing.scale(X, axis=0, with_mean=True, with_std=True, copy=False)             # scale data 
X = X.astype(np.float32)

y = boston['target']
y = y.astype(np.float32)

random.seed(1)
spl = np.random.choice(X.shape[0], int(0.75 * X.shape[0]), replace = False)              # split data
not_spl = np.array(list(set(np.array(range(X.shape[0]))) - set(spl)))

print len(spl), len(not_spl)

Xtr = X[spl]
Xte = X[not_spl]

y_tr = y[spl]
y_te = y[not_spl]


# train regression model

# initialize
init = Regression(Xtr, y_tr, Xte, y_te, iters = 10000, learning_rate = 0.5, optimizer = 'sgd', batch_size = None, 
                  
                  L1 = 0.0001, L2 = 0.0001, maximize = False, early_stopping_rounds = 10, weights_initialization = 'glorot_uniform', 
                  
                  objective = 'mean_squared_error', linear_regression = True, add_bias = True, custom_eval = None)

# fit
fit = init.fit()

...
# iter 223   train_loss  0.539   test_loss  171.277
# iter 224   train_loss  0.537   test_loss  170.423
# iter 225   train_loss  0.535   test_loss  169.574
# iter 226   train_loss  0.532   test_loss  168.73
# iter 227   train_loss  0.53   test_loss  167.89
 ...
# iter 1239   train_loss  0.126   test_loss  13.229
# iter 1240   train_loss  0.126   test_loss  13.225
# iter 1241   train_loss  0.126   test_loss  13.222
# iter 1242   train_loss  0.126   test_loss  13.218
# iter 1243   train_loss  0.126   test_loss  13.214
 ...
# iter 1739   train_loss  0.124   test_loss  12.747
# iter 1740   train_loss  0.124   test_loss  12.747
# iter 1741   train_loss  0.124   test_loss  12.747
# iter 1742   train_loss  0.124   test_loss  12.747
# iter 1743   train_loss  0.124   test_loss  12.747
# regression stopped after  10  consecutive  increases  of loss and  1743  Epochs


# predictions for train, test
 
pred_tr = init.PREDICT(Xtr)
pred_te = init.PREDICT(Xte)

# mse train - test
print 'mean_squared_error on train data is', str(metrics.mean_squared_error(y_tr, pred_tr))
# mean_squared_error on train data is 27.0498

print 'mean_squared_error on test data is', str(metrics.mean_squared_error(y_te, pred_te))
# mean_squared_error on test data is 12.7468
```
<br>
For a random train-test split on the data the sgd-linear-regression gives after 1740 iterations a mse-error of 27.04 on train-data and 12.74 on test-data. As a comparison I've run also a sklearn-linear-regression, which gives comparable results,
<br>

```py
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(Xtr, y_tr)


pred_tr_skl = lr.predict(Xtr)
pred_te_skl = lr.predict(Xte)

# mse train - test
print 'mean_squared_error on train data is', str(metrics.mean_squared_error(y_tr, pred_tr_skl))
# mean_squared_error on train data is 24.8352

print 'mean_squared_error on test data is', str(metrics.mean_squared_error(y_te, pred_te_skl))
# mean_squared_error on test data is 14.51736
```

<br>
A few words about the parameters in the **Regression()** function,

parameter               | notes
----------              | -----
X                       | train data 
Y                       | train response 
X_test                  | test data 
Y_test                  | test response 
iters                   | number of iterations
learning_rate           | learning rate
optimizer               | 'sgd'
batch_size              | if None then in each iteration all data will be used for training, if integer value then a mini-batch will be used for training
L1                      | L1-regularization        
L2                      | L2-regularization
maximize                | if the evaluation metric should be maximized/minimized (used in early-stopping)
early_stopping_rounds   | after how many iters of increasing/decreasing loss should the training-process stop
weights_initialization  | one of 'uniform', 'normal', 'glorot_uniform'
objective               | the objective function
linear_regression       | boolean [True, False] (linear_regression or logistic_regression)
add_bias                | if bias should be added to the model
custom_eval             | use a custom evaluation function in form of a tuple (function, 'name_of_function') to evaluate the test data during training, otherwise None

<br>
The second example is about classification using the **mnist** data set (digit recognition),

```py

from Regression_theano import Regression
from sklearn import preprocessing
import pandas as pd
import random
from sklearn import metrics
from scikits.statsmodels.tools import categorical


df = pd.read_csv('train.csv')                                       # assuming the train data is downloaded in the current working directory
 
X = np.array(df.iloc[:,1:df.shape[1]].as_matrix(columns=None)) 
X = X.astype(np.float32)
X /= 255                                                          # divide pixels to the maximum to get values between 0 and 1

target = np.array(df.iloc[:, 0].as_matrix(columns=None))


random.seed(1)
spl = np.random.choice(X.shape[0], int(0.75 * X.shape[0]), replace = False)       # split data
not_spl = np.array(list(set(np.array(range(X.shape[0]))) - set(spl)))

Xtr = X[spl]
Xte = X[not_spl]

y_tr = target[spl]
y_te = target[not_spl]

y_categ = categorical(y_tr, drop = True)                              # response should be one-hot-encoded
y_categ_te = categorical(y_te, drop = True)

print Xtr.shape, Xte.shape, y_categ.shape, y_categ_te.shape      
# (31500, 784) (10500, 784) (31500, 10) (10500, 10)



def evaluation_func(y_test, pred_test):                                                      # customized evaluation metric [ accuracy ]
    
    out = np.mean(np.argmax(pred_test, axis = 1) == np.argmax(y_test, axis = 1))
    
    return out


# initialize data

init = Regression(Xtr, y_categ, Xte, y_categ_te, iters = 1000, learning_rate = 0.5, optimizer = 'sgd', batch_size = 512, 
                  
                  L1 = 0.00001, L2 = 0.00001, maximize = False, early_stopping_rounds = 10, weights_initialization = 'normal', 
                  
                  objective = 'categorical_crossentropy', linear_regression = False, add_bias = True, custom_eval = (evaluation_func, 'accuracy'))
    
# fit data

fit = init.fit()

# iter 1   train_loss  0.497   test_accuracy   0.877
# iter 2   train_loss  0.441   test_accuracy   0.894
# iter 3   train_loss  0.418   test_accuracy   0.901
# iter 4   train_loss  0.404   test_accuracy   0.904
# iter 5   train_loss  0.394   test_accuracy   0.906
# iter 6   train_loss  0.386   test_accuracy   0.907
# iter 7   train_loss  0.38    test_accuracy   0.909
# iter 8   train_loss  0.374   test_accuracy   0.91
# iter 9   train_loss  0.37    test_accuracy   0.911
# iter 10  train_loss  0.366   test_accuracy   0.912
# iter 11  train_loss  0.363   test_accuracy   0.913
# regression stopped after  10  consecutive  increases  of loss and  11  Epochs

# predictions

pred_tr = init.PREDICT(Xtr)
pred_te = init.PREDICT(Xte)


# accuracy train - test

print 'mean_squared_error on train data is', str(metrics.accuracy_score(y_tr, np.argmax(pred_tr, axis = 1)))
# mean_squared_error on train data is 0.9173

print 'mean_squared_error on test data is', str(metrics.accuracy_score(y_te, np.argmax(pred_te, axis = 1)))
# mean_squared_error on test data is 0.9123

```
<br>
Nowadays, digit recognition using convolutional neural networks approaches [0.0 %](http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#4d4e495354), so I think that approximately 91.0 % accuracy using a single sgd-logistic-regression model is not that bad.
<br>
<br>

#### final word

A linear or logistic regression model in theano can be thought of as a neural network with a single hidden layer. It can be used as a basis to build a neural network by adding, for instance, a certain number of hidden layers, dropout or batch-normalization.    
