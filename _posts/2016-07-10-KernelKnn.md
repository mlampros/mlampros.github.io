---
layout: post
title: Kernel k nearest neighbors
tags: [R, package, R-bloggers]
comments: true
---


This blog post is about my recently released package on CRAN, **KernelKnn**. The package consists of three functions **KernelKnn**, **KernelKnnCV** and **knn.index.dist**. It also includes two data sets (*housing data*, *ionosphere*), which will be used here to illustrate the functionality of the package.


### k nearest neighbors

In pattern recognition the [k nearest neighbors](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) (KNN) is a non-parametric method used for classification and regression. Although KNN belongs to the 10 most influential algorithms in data mining, it is considered as one of the simplest in machine learning. 
<br><br>
The most important parameters of the KNN algorithm are **k** and the **distance metric**. The parameter k specifies the number of neighbor observations that contribute to the output predictions. Optimal values for k can be obtained mainly through resampling methods, such as *cross-validation* or *bootstrap*. The distance metric is another important factor, which depending on the data set can affect the performance of the algorithm. Widely used distance metrics are the *euclidean*, *manhattan*, *chebyshev*, *minkowski* and *hamming*.
<br><br>
The simple KNN algorithm can be extended by giving different weights to the selected k nearest neighbors. A common weighting scheme consists in giving each neighbor a weight of 1/d, where d is the distance to the neighbor. The purpose of the *KernelKnn* package is to use different weight functions (kernels) in order to optimize the output predictions in both regression and classification.


### KernelKnn function

The following code snippets appear in the package Vignettes. I'll illustrate the package functionality using mainly classification and in-between I'll point out the differences for regression and classification, which could lead to potential errors.


```R

data(ionosphere, package = 'KernelKnn')

apply(ionosphere, 2, function(x) length(unique(x)))

##    V1    V2    V3    V4    V5    V6    V7    V8    V9   V10   V11   V12 
##     2     1   219   269   204   259   231   260   244   267   246   269 
##   V13   V14   V15   V16   V17   V18   V19   V20   V21   V22   V23   V24 
##   238   266   234   270   254   280   254   266   248   265   248   264 
##   V25   V26   V27   V28   V29   V30   V31   V32   V33   V34 class 
##   256   273   256   281   244   266   243   263   245   263     2

```


```R

# the second column will be removed as it has a single unique value

ionosphere = ionosphere[, -2]


```
<br>

When using an algorithm where the output depends on distance calculation (as is the case in k-nearest-neighbors) it is recommended to first scale the data,
<br><br>

```R

X = scale(ionosphere[, -ncol(ionosphere)])
y = ionosphere[, ncol(ionosphere)]


```
<br>

Both in regression and classification the **KernelKnn** function accepts a numeric vector as a response variable (here y). In **classification** the user should additionally give the unique values of the labels, which should range from 1:Inf. This is important otherwise the internal functions do not work. The *KernelKnn* function returns a vector of numeric values in case of regression or a matrix of class probabilities in case of classification.
<br><br>

```R
# convert y from factor to numeric in classification

y = c(1:length(unique(y)))[ match(ionosphere$class, sort(unique(ionosphere$class))) ]


# random split in train-test

spl_train = sample(1:length(y), round(length(y) * 0.75))

spl_test = setdiff(1:length(y), spl_train)

```

```R

str(spl_train)

```

```R

##  int [1:263] 56 10 224 249 109 223 221 146 93 194 

```

```R

str(spl_test)

```

```R

##  int [1:88] 2 4 7 9 11 15 20 23 33 34 ...

```

```R
# evaluation metric

acc = function (y_true, preds) {
  
  out = table(y_true, max.col(preds, ties.method = "random"))
  
  acc = sum(diag(out))/sum(out)
  
  acc
}

```
<br>


A simple k-nearest-neighbors model can be run with weights_function = NULL and the parameter ‘regression’ should be set to FALSE in case of classification.
<br>

```R
library(KernelKnn)

preds_TEST = KernelKnn(X[spl_train, ], TEST_data = X[spl_test, ], y[spl_train], k = 5 , 
                       
                       method = 'euclidean', weights_function = NULL, regression = F,
                       
                       Levels = unique(y))

```

```R

head(preds_TEST)

```


```R
##      class_1 class_2
## [1,]       0       1
## [2,]       0       1
## [3,]       0       1
## [4,]       0       1
## [5,]       0       1
## [6,]       0       1

```
<br>

K-nearest-neigbor calculations in the KernelKnn function can be accomplished using the following distance metrics : *euclidean*, *manhattan*, *chebyshev*, *canberra*, *braycurtis*, *minkowski* (by default the order ‘p’ of the minkowski parameter equals k), *hamming*, *mahalanobis*, *pearson_correlation*, *simple_matching_coefficient*, *jaccard_coefficient* and *Rao_coefficient*. The last four are similarity measures and are appropriate for binary data [0,1]. 
<br>

There are two ways to use a kernel in the KernelKnn function. The **first option** is to choose one of the existing kernels (*uniform*, *triangular*, *epanechnikov*, *biweight*, *triweight*, *tricube*, *gaussian*, *cosine*, *logistic*, *silverman*, *inverse*, *gaussianSimple*, *exponential*). Here, I'll use the canberra metric and the tricube kernel because they give optimal results (according to my RandomSearchR package),

```R
preds_TEST_tric = KernelKnn(X[spl_train, ], TEST_data = X[spl_test, ], y[spl_train], k = 10 , 
                            
                            method = 'canberra', weights_function = 'tricube', regression = F,  
                            
                            Levels = unique(y))
head(preds_TEST_tric)

```


```R
##              [,1]       [,2]
## [1,] 1.745564e-02 0.98254436
## [2,] 9.667304e-01 0.03326963
## [3,] 0.000000e+00 1.00000000
## [4,] 6.335040e-18 1.00000000
## [5,] 4.698239e-02 0.95301761
## [6,] 0.000000e+00 1.00000000

```
<br>

The **second option** is to give a self-defined kernel function. Here, I’ll pick the density function of the normal distribution with mean = 0.0 and standard deviation = 1.0 (the data are scaled to have mean zero and unit variance),

```R
norm_kernel = function(W) {
  
  W = dnorm(W, mean = 0, sd = 1.0)
  
  W = W / rowSums(W)
  
  return(W)
}


preds_TEST_norm = KernelKnn(X[spl_train, ], TEST_data = X[spl_test, ], y[spl_train], k = 10 , 
                            
                            method = 'canberra', weights_function = norm_kernel, regression = F, 
                            
                            Levels = unique(y))
head(preds_TEST_norm)

```


```R
##            [,1]      [,2]
## [1,] 0.26150003 0.7385000
## [2,] 0.84170089 0.1582991
## [3,] 0.00000000 1.0000000
## [4,] 0.07614579 0.9238542
## [5,] 0.09479386 0.9052061
## [6,] 0.00000000 1.0000000

```
<br>

The computations can be speed up by using the parameter **threads** (utilizes openMP). There is also the option to exclude **extrema** (minimum and maximum distances) during the calculation of the k-nearest-neighbor distances using extrema = TRUE. The bandwidth of the existing kernels can be tuned using the **h** parameter, which defaults to 1.0. 
<br><br><br>

### The KernelKnnCV function

<br>

The *KernelKnnCV* function can be employed to return the prediction accuracy using n-fold cross-validation. The following parameter pairs give optimal results according to my RandomSearchR package,


| k  |      method    |  kernel       |
|----|:--------------:|:--------------|
| 10 |  canberra      | tricube       |
| 9  |    canberra    |  epanechnikov |


<br>


```R

fit_cv_pair1 = KernelKnnCV(X, y, k = 10 , folds = 5, method = 'canberra', 
                           
                           weights_function = 'tricube', regression = F, 
                           
                           Levels = unique(y), threads = 5)

```

```R

str(fit_cv_pair1)

```

```R

## List of 2
##  $ preds:List of 5
##   ..$ : num [1:71, 1:2] 0.00648 0.25323 1 0.97341 0.92031 ...
##   ..$ : num [1:70, 1:2] 0 0 0 0 0.999 ...
##   ..$ : num [1:70, 1:2] 0.353 0 0.17 0.212 0.266 ...
##   ..$ : num [1:70, 1:2] 0 0 0 0 0 ...
##   ..$ : num [1:70, 1:2] 0.989 0 1 0 0 ...
##  $ folds:List of 5
##   ..$ fold_1: int [1:71] 5 26 233 243 30 41 237 229 19 11 ...
##   ..$ fold_2: int [1:70] 262 89 257 67 58 266 253 85 275 268 ...
##   ..$ fold_3: int [1:70] 127 128 295 287 134 288 130 277 125 101 ...
##   ..$ fold_4: int [1:70] 313 301 317 318 316 142 175 157 146 147 ...
##   ..$ fold_5: int [1:70] 195 326 225 332 342 347 206 219 218 214 ...

```

```R

fit_cv_pair2 = KernelKnnCV(X, y, k = 9 , folds = 5,method = 'canberra',
                           
                           weights_function = 'epanechnikov', regression = F,
                           
                           Levels = unique(y), threads = 5)

```


```R

str(fit_cv_pair2)

```


```R

## List of 2
##  $ preds:List of 5
##   ..$ : num [1:71, 1:2] 0.0224 0.255 1 0.9601 0.8876 ...
##   ..$ : num [1:70, 1:2] 0 0 0 0 0.998 ...
##   ..$ : num [1:70, 1:2] 0.36 0 0.164 0.185 0.202 ...
##   ..$ : num [1:70, 1:2] 0 0 0 0 0 ...
##   ..$ : num [1:70, 1:2] 0.912 0 1 0 0 ...
##  $ folds:List of 5
##   ..$ fold_1: int [1:71] 5 26 233 243 30 41 237 229 19 11 ...
##   ..$ fold_2: int [1:70] 262 89 257 67 58 266 253 85 275 268 ...
##   ..$ fold_3: int [1:70] 127 128 295 287 134 288 130 277 125 101 ...
##   ..$ fold_4: int [1:70] 313 301 317 318 316 142 175 157 146 147 ...
##   ..$ fold_5: int [1:70] 195 326 225 332 342 347 206 219 218 214 ...

```
<br>

Each cross-validated object returns a list of length 2 ( the first sublist includes the predictions for each fold whereas the second gives the indices of the folds)

<br>

```R
acc_pair1 = unlist(lapply(1:length(fit_cv_pair1$preds), 
                          
                          function(x) acc(y[fit_cv_pair1$folds[[x]]], 
                                          
                                          fit_cv_pair1$preds[[x]])))
acc_pair1

```


```R

## [1] 0.9154930 0.9142857 0.9142857 0.9285714 0.9571429

```


```R

cat('accurcay for params_pair1 is :', mean(acc_pair1), '\n')

```


```R

## accurcay for params_pair1 is : 0.9259557

```


```R

acc_pair2 = unlist(lapply(1:length(fit_cv_pair2$preds), 
                          
                          function(x) acc(y[fit_cv_pair2$folds[[x]]], 
                                          
                                          fit_cv_pair2$preds[[x]])))
acc_pair2

```


```R

## [1] 0.9014085 0.9142857 0.9000000 0.9142857 0.9571429

```

```R

cat('accuracy for params_pair2 is :', mean(acc_pair2), '\n')

```

```R

## accuracy for params_pair2 is : 0.9174245

```
<br><br>

### Adding or multiplying kernels

<br>

In the KernelKnn package the user can also combine kernels by adding or multiplying from the existing ones. For instance, if I want to multiply the tricube with the gaussian kernel, then I’ll give the following character string to the weights_function, *“tricube_gaussian_MULT”*. On the other hand, If I want to add the same kernels then the weights_function will be *“tricube_gaussian_ADD”*. 
<br>

I'll illustrate this option of the package using two image data sets in form of matrices, i.e. the *MNIST* and the *CIFAR-10*. From within R one can download the data in a linux OS using,

```R
system(“wget https://raw.githubusercontent.com/mlampros/DataSets/master/mnist.zip”)

and 

system(“wget https://raw.githubusercontent.com/mlampros/DataSets/master/cifar_10.zip”)

```
<br>

Moreover, the **irlba** and the **OpenImageR** packages are needed for comparison purposes, which can be installed directly from CRAN using the install.packages() function. A 4-fold cross-validation using the KernelKnnCV function can take 40-50 minutes utilizing 6 threads (for each data set). An alternative to reduce the computation time would be a train-test split of the data at the cost of performance validation.

<br>

##### **MNIST data set**

The MNIST data set of handwritten digits has a training set of 70,000 examples and each row of the matrix corresponds to a 28 x 28 image. The unique values of the response variable *y* range from 0 to 9. More information about the data can be found in the [*DataSets* ](https://github.com/mlampros/DataSets) repository (the folder includes also an Rmarkdown file).


```R

system("wget https://raw.githubusercontent.com/mlampros/DataSets/master/mnist.zip")             

mnist <- read.table(unz("mnist.zip", "mnist.csv"), nrows = 70000, header = T, 
                    
                    quote = "\"", sep = ",")

```


```R
X = mnist[, -ncol(mnist)]
dim(X)

## [1] 70000   784

# the KernelKnn function requires that the labels are numeric and start from 1 : Inf

y = mnist[, ncol(mnist)] + 1          
table(y)

## y
##    1    2    3    4    5    6    7    8    9   10 
## 6903 7877 6990 7141 6824 6313 6876 7293 6825 6958

```
<br>

K nearest neighbors do not perform well in high dimensions due to the *curse of dimensionality* (k observations that are nearest to a given test observation x1 may be very far away from x1 in p-dimensional space when p is large [ An introduction to statistical learning, James/Witten/Hastie/Tibshirani, pages 108-109 ]), leading to a very poor k-nearest-neighbors fit. One option to overcome this problem would be to use truncated svd (irlba package) to reduce the dimensions of the data. A second option, which is appropriate in case of images, would be to use image descriptors. Here, I'll compare those two approaches. <br><br><br>

##### **KernelKnnCV using truncated svd**
<br>

I experimented with different settings and the following parameters gave the best results,
<br><br>


|irlba_singlular_vectors | k  |      method    |  kernel                     |
|:-----------------------|----|:--------------:|:----------------------------|
|40                      | 8  |  braycurtis    | biweight_tricube_MULT       |


<br>

```R

library(irlba)

svd_irlb = irlba(as.matrix(X), nv = 40, nu = 40, verbose = F)            # irlba truncated svd

new_x = as.matrix(X) %*% svd_irlb$v               # new_x using the 40 right singular vectors

```
<br>

```R

fit = KernelKnnCV(as.matrix(new_x), y, k = 8, folds = 4, method = 'braycurtis',
                  
                  weights_function = 'biweight_tricube_MULT', regression = F, 
                  
                  threads = 6, Levels = sort(unique(y)))


# str(fit)

```
<br>

```R

acc_fit = unlist(lapply(1:length(fit$preds), 
                        
                        function(x) acc(y[fit$folds[[x]]], 
                                        
                                        fit$preds[[x]])))
acc_fit

## [1] 0.9742857 0.9749143 0.9761143 0.9741143

cat('mean accuracy using cross-validation :', mean(acc_fit), '\n')

## mean accuracy using cross-validation : 0.9748571

```
<br>

Utilizing truncated svd a 4-fold cross-validation KernelKnn model gives a 97.48% accuracy.
<br><br><br>

##### **KernelKnnCV and HOG (histogram of oriented gradients)**
<br>

In this chunk of code, besides KernelKnnCV I'll also use HOG. The histogram of oriented gradients (HOG) is a feature descriptor used in computer vision and image processing for the purpose of object detection. The technique counts occurrences of gradient orientation in localized portions of an image. This method is similar to that of edge orientation histograms, scale-invariant feature transform descriptors, and shape contexts, but differs in that it is computed on a dense grid of uniformly spaced cells and uses overlapping local contrast normalization for improved accuracy (Wikipedia).
<br>

```R

library(OpenImageR)

hog = HOG_apply(X, cells = 6, orientations = 9, height = 28, width = 28, threads = 6)

## 
## time to complete : 1.802997 secs

dim(hog)

## [1] 70000   324

```
<br>

```R

fit_hog = KernelKnnCV(hog, y, k = 20, folds = 4, method = 'braycurtis',
                  
                  weights_function = 'biweight_tricube_MULT', regression = F, 
                  
                  threads = 6, Levels = sort(unique(y)))


#str(fit_hog)

```
<br>

```R

acc_fit_hog = unlist(lapply(1:length(fit_hog$preds), 
                            
                            function(x) acc(y[fit_hog$folds[[x]]], 
                                            
                                            fit_hog$preds[[x]])))
acc_fit_hog

## [1] 0.9833714 0.9840571 0.9846857 0.9838857

cat('mean accuracy for hog-features using cross-validation :', mean(acc_fit_hog), '\n')

## mean accuracy for hog-features using cross-validation : 0.984

```
<br>

By changing from the simple svd-features to HOG-features the accuracy of a 4-fold cross-validation model increased from 97.48% to 98.4% (approx. 1% difference)

<br><br><br>

##### **CIFAR-10 data set**
<br>

CIFAR-10 is an established computer-vision dataset used for object recognition. The data I'll use in this example is a subset of an 80 million tiny images dataset and consists of 60,000 32x32 color images containing one of 10 object classes ( 6000 images per class ). Furthermore, the data were converted from RGB to gray, normalized and rounded to 2 decimal places (to reduce the storage size). More information about the data can be found in the [*DataSets*](https://github.com/mlampros/DataSets) repository (the folder includes also an Rmarkdown file).
<br><br>

I'll build the kernel k-nearest-neighbors models in the same way I've done for the mnist data set and then I'll compare the results.


```R

system("wget https://raw.githubusercontent.com/mlampros/DataSets/master/cifar_10.zip")      

cifar_10 <- read.table(unz("cifar_10.zip", "cifar_10.csv"), nrows = 60000, header = T, 
                       
                       quote = "\"", sep = ",")

```
<br><br>

##### **KernelKnnCV using truncated svd**
<br>

```R
X = cifar_10[, -ncol(cifar_10)]
dim(X)

## [1] 60000  1024

# the KernelKnn function requires that the labels are numeric and start from 1 : Inf

y = cifar_10[, ncol(cifar_10)]         
table(y)

## y
##    1    2    3    4    5    6    7    8    9   10 
## 6000 6000 6000 6000 6000 6000 6000 6000 6000 6000

```
<br>


The parameter settings are similar to those of the mnist data,

<br>

```R

svd_irlb = irlba(as.matrix(X), nv = 40, nu = 40, verbose = F)            # irlba truncated svd

new_x = as.matrix(X) %*% svd_irlb$v               # new_x using the 40 right singular vectors

```
<br>

```R

fit = KernelKnnCV(as.matrix(new_x), y, k = 8, folds = 4, method = 'braycurtis',
                  
                  weights_function = 'biweight_tricube_MULT', regression = F,
                  
                  threads = 6, Levels = sort(unique(y)))


# str(fit)

```
<br>

```R

acc_fit = unlist(lapply(1:length(fit$preds),
                        
                        function(x) acc(y[fit$folds[[x]]], 
                                        
                                        fit$preds[[x]])))
acc_fit

## [1] 0.4080667 0.4097333 0.4040000 0.4102667

cat('mean accuracy using cross-validation :', mean(acc_fit), '\n')

## mean accuracy using cross-validation : 0.4080167

```
<br>

The accuracy of a 4-fold cross-validation model using truncated svd is 40.8%.

<br><br><br>

##### **KernelKnnCV using HOG (histogram of oriented gradients)**

<br>
Next, I'll run the KernelKnnCV using the HOG-descriptors,
<br><br>


```R

hog = HOG_apply(X, cells = 6, orientations = 9, height = 32,
                
                width = 32, threads = 6)

## 
## time to complete : 3.394621 secs

dim(hog)

## [1] 60000   324

```
<br>

```R

fit_hog = KernelKnnCV(hog, y, k = 20, folds = 4, method = 'braycurtis',
                  
                  weights_function = 'biweight_tricube_MULT', regression = F,
                  
                  threads = 6, Levels = sort(unique(y)))


# str(fit_hog)

```
<br>

```R

acc_fit_hog = unlist(lapply(1:length(fit_hog$preds), 
                            
                            function(x) acc(y[fit_hog$folds[[x]]], 
                                            
                                            fit_hog$preds[[x]])))
acc_fit_hog

## [1] 0.5807333 0.5884000 0.5777333 0.5861333

cat('mean accuracy for hog-features using cross-validation :', mean(acc_fit_hog), '\n')

## mean accuracy for hog-features using cross-validation : 0.58325

```
<br>

By using hog-descriptors in a 4-fold cross-validation model the accuracy in the cifar-10 data increases from 40.8% to 58.3% (approx. 17.5% difference).  


<br><br>


An updated version of the **KernelKnn** package can be found in the [Github repository](https://github.com/mlampros/KernelKnn) and to report bugs/issues please use the following link, [https://github.com/mlampros/KernelKnn/issues](https://github.com/mlampros/KernelKnn/issues).
