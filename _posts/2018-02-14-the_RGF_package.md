---
layout: post
title: Regularized Greedy Forest in R
tags: [R, package, R-bloggers]
comments: true
---


This blog post is about my newly released [RGF package](https://github.com/mlampros/RGF) (the blog post consists mainly of the package Vignette). The *RGF* package is a wrapper of the [*Regularized Greedy Forest*](https://github.com/fukatani/rgf_python) *python* package, which also includes a [*Multi-core implementation (FastRGF)*](https://github.com/baidu/fast_rgf). Portability from Python to R was made possible using the [reticulate](https://github.com/rstudio/reticulate) package and the installation requires basic knowledge of Python. Except for the Linux Operating System, the installation on Macintosh and Windows might be somehow cumbersome (*on windows the package currently can be used only from within the command prompt*). Detailed installation instructions for all three Operating Systems can be found in the *README.md* file and in the [*rgf_python*](https://github.com/fukatani/rgf_python) Github repository.

<br>


**UPDATE 26-07-2018**: A [Singularity image file](http://mlampros.github.io/2018/07/26/singularity_containers/) is available in case that someone intends to run *RGF* on Ubuntu Linux (locally or in a cloud instance) with all package requirements pre-installed. This allows the user to utilize the *RGF* package without having to spend time on the installation process.

<br>

The *Regularized Greedy Forest* algorithm is explained in detail in the paper [*Rie Johnson and Tong Zhang, Learning Nonlinear Functions Using Regularized Greedy Forest*](https://arxiv.org/abs/1109.0887). A small synopsis would be *"... the resulting method, which we refer to as regularized greedy forest (RGF), integrates two ideas: one is to include tree-structured regularization into the learning formulation; and the other is to employ the fully-corrective regularized greedy algorithm ...."*.

<br>

At the time of writing this blog post (14 - 02 - 2018), there isn't a corresponding implementation of the algorithm in the R language, so I decided to port the Python package in R taking advantage of the reticulate package. In the next lines, I will explain the functionality of the package and I compare RGF with other similar implementations, such as [*ranger*](https://github.com/imbs-hl/ranger) (random forest algorithm) and [*xgboost*](https://github.com/dmlc/xgboost/tree/master/R-package) (gradient boosting algorithm), in terms of time efficiency and error rate improvement.

<br>

#### **The RGF package**

<br> 

The *RGF* package includes the following R6-classes / functions,

<br>

##### **classes** 


<br>


|    RGF_Regressor        |   RGF_Classifier       |  FastRGF_Regressor    |  FastRGF_Classifier  |
| :-----------------:     |  :-----------------:   | :------------------:  | :------------------: |
|   fit()                 |  fit(()                | fit()                 | fit()                |
|   predict()             |  predict()             | predict()             | predict()            |
|   cleanup()             |  predict_proba()       | cleanup()             | predict_proba()      |
|   get_params()          |  cleanup()             | get_params()          | cleanup()            |
|   score()               |  get_params()          | score()               | get_params()         |
|   feature_importances() |  score()               |                       | score()              |
|   dump_model()          |  feature_importances() |                       |                      |
|                         |  dump_model()          |                       |                      |  


<br>
  
  
##### **functions**

**UPDATE 10-05-2018** : Beginning from version **1.0.3** the **dgCMatrix_2scipy_sparse** function was renamed to **TO_scipy_sparse** and now accepts either a *dgCMatrix* or a *dgRMatrix* as input. The appropriate format for the RGF package in case of sparse matrices is the **dgCMatrix** format (*scipy.sparse.csc_matrix*) 

<br>

TO_scipy_sparse()

RGF_cleanup_temp_files()

mat_2scipy_sparse()

<br>


The package documentation includes details and examples for all R6-classes and functions. In the following code chunks, I'll explain how a user can work with sparse matrices as all RGF algorithms (besides a dense matrix) **require a python sparse matrix as input**. 

<br>


#### **Sparse matrices as input**

<br>

The RGF package includes two functions (**mat_2scipy_sparse** and **TO_scipy_sparse**) which allow the user to convert from a *matrix* / *sparse matrix* (*dgCMatrix*, *dgRMatrix*) to a *scipy sparse matrix* (*scipy.sparse.csc_matrix*, *scipy.sparse.csr_matrix*),

<br>

```R

library(nmslibR)

# conversion from a matrix object to a scipy sparse matrix
#----------------------------------------------------------

set.seed(1)

x = matrix(runif(1000), nrow = 100, ncol = 10)

x_sparse = mat_2scipy_sparse(x, format = "sparse_row_matrix")

print(dim(x))

[1] 100  10

print(x_sparse$shape)

(100, 10)
  
```

<br>


```R

# conversion from a dgCMatrix object to a scipy sparse matrix
#-------------------------------------------------------------

data = c(1, 0, 2, 0, 0, 3, 4, 5, 6)


# 'dgCMatrix' sparse matrix
#--------------------------

dgcM = Matrix::Matrix(data = data, nrow = 3,

                      ncol = 3, byrow = TRUE,

                      sparse = TRUE)

print(dim(dgcM))

[1] 3 3

x_sparse = TO_scipy_sparse(dgcM)

print(x_sparse$shape)

(3, 3)


# 'dgRMatrix' sparse matrix
#--------------------------

dgrM = as(dgcM, "RsparseMatrix")

class(dgrM)

# [1] "dgRMatrix"
# attr(,"package")
# [1] "Matrix"

print(dim(dgrM))

[1] 3 3

res_dgr = TO_scipy_sparse(dgrM)

print(res_dgr$shape)

(3, 3)
  
```


<br>



#### **Comparison of RGF with ranger and xgboost**

<br>

First the data, libraries and cross-validation function will be inputted (the *MLmetrics* library is also required),

<br>

```R

data(Boston, package = 'KernelKnn')

library(RGF)
library(ranger)
library(xgboost)



# shuffling function for cross-validation folds
#-----------------------------------------------


func_shuffle = function(vec, times = 10) {

  for (i in 1:times) {

    out = sample(vec, length(vec))
  }
  out
}


# cross-validation folds [ regression]
#-------------------------------------


regr_folds = function(folds, RESP, stratified = FALSE) {

  if (is.factor(RESP)) {

    stop(simpleError("this function is meant for regression for classification use the 'class_folds' function"))
  }

  samp_vec = rep(1/folds, folds)

  sort_names = paste0('fold_', 1:folds)

  if (stratified == TRUE) {

    stratif = cut(RESP, breaks = folds)

    clas = lapply(unique(stratif), function(x) which(stratif == x))

    len = lapply(clas, function(x) length(x))

    prop = lapply(len, function(y) sapply(1:length(samp_vec), function(x) round(y * samp_vec[x])))

    repl = unlist(lapply(prop, function(x) sapply(1:length(x), function(y) rep(paste0('fold_', y), x[y]))))

    spl = suppressWarnings(split(1:length(RESP), repl))}

  else {

    prop = lapply(length(RESP), function(y) sapply(1:length(samp_vec), function(x) round(y * samp_vec[x])))

    repl = func_shuffle(unlist(lapply(prop, function(x) sapply(1:length(x), function(y) rep(paste0('fold_', y), x[y])))))

    spl = suppressWarnings(split(1:length(RESP), repl))
  }

  spl = spl[sort_names]

  if (length(table(unlist(lapply(spl, function(x) length(x))))) > 1) {

    warning('the folds are not equally split')
  }

  if (length(unlist(spl)) != length(RESP)) {

    stop(simpleError("the length of the splits are not equal with the length of the response"))
  }

  spl
}
```

<br>

#### **single threaded    ( small data set )**



<br>

In the next code chunk, I'll perform 5-fold cross-validation using the Boston dataset and I'll compare time execution and error rate for all three algorithms (comparison **without doing hyper-parameter tuning**),

<br>

```R

NUM_FOLDS = 5

set.seed(1)
FOLDS = regr_folds(folds = NUM_FOLDS, Boston[, 'medv'], stratified = T)


boston_rgf_te = boston_ranger_te = boston_xgb_te = boston_rgf_time = boston_ranger_time = boston_xgb_time = rep(NA, NUM_FOLDS)


for (i in 1:length(FOLDS)) {

  cat("fold : ", i, "\n")

  samp = unlist(FOLDS[-i])
  samp_ = unlist(FOLDS[i])


  # RGF
  #----

  rgf_start = Sys.time()

  init_regr = RGF_Regressor$new(l2 = 0.1)

  init_regr$fit(x = as.matrix(Boston[samp, -ncol(Boston)]), y = Boston[samp, ncol(Boston)])

  pr_te = init_regr$predict(as.matrix(Boston[samp_, -ncol(Boston)]))

  rgf_end = Sys.time()

  boston_rgf_time[i] = rgf_end - rgf_start

  boston_rgf_te[i] = MLmetrics::RMSE(Boston[samp_, 'medv'], pr_te)


  # ranger
  #-------

  ranger_start = Sys.time()

  fit = ranger(dependent.variable.name = "medv", data = Boston[samp, ], write.forest = TRUE, 
               
               probability = F, num.threads = 1, num.trees = 500, verbose = T, 
               
               classification = F, mtry = NULL, min.node.size = 5, keep.inbag = T)

  pred_te = predict(fit, data = Boston[samp_, -ncol(Boston)], type = 'se')$predictions

  ranger_end = Sys.time()

  boston_ranger_time[i] = ranger_end - ranger_start

  boston_ranger_te[i] = MLmetrics::RMSE(Boston[samp_, 'medv'], pred_te)


  # xgboost
  #--------

  xgb_start = Sys.time()

  dtrain <- xgb.DMatrix(data = as.matrix(Boston[samp, -ncol(Boston)]), label = Boston[samp, ncol(Boston)])

  dtest <- xgb.DMatrix(data = as.matrix(Boston[samp_, -ncol(Boston)]), label = Boston[samp_, ncol(Boston)])

  
  watchlist <- list(train = dtrain, test = dtest)

  
  param = list("objective" = "reg:linear", "bst:eta" = 0.05, "max_depth" = 4, 
               
               "subsample" = 0.85, "colsample_bytree" = 0.85, "booster" = "gbtree",
               
               "nthread" = 1)

  fit = xgb.train(param, dtrain, nround = 500, print_every_n  = 100, watchlist = watchlist, early_stopping_rounds = 20,
                  
                  maximize = FALSE, verbose = 0)

  p_te = xgboost:::predict.xgb.Booster(fit, as.matrix(Boston[samp_, -ncol(Boston)]), ntreelimit = fit$best_iteration)

  xgb_end = Sys.time()

  boston_xgb_time[i] = xgb_end - xgb_start

  boston_xgb_te[i] = MLmetrics::RMSE(Boston[samp_, 'medv'], p_te)
}

```

<br>

```R

fold :  1 
fold :  2 
fold :  3 
fold :  4 
fold :  5 

```

<br>

```R

cat("total time rgf 5 fold cross-validation : ", sum(boston_rgf_time), " mean rmse on test data : ", mean(boston_rgf_te), "\n")

cat("total time ranger 5 fold cross-validation : ", sum(boston_ranger_time), " mean rmse on test data : ", mean(boston_ranger_te), "\n")

cat("total time xgb 5 fold cross-validation : ", sum(boston_xgb_time), " mean rmse on test data : ", mean(boston_xgb_te), "\n")

```

<br>

```R

total time rgf 5 fold cross-validation :  0.7730639  mean rmse on test data :  3.832135 
total time ranger 5 fold cross-validation :  3.826846  mean rmse on test data :  4.17419 
total time xgb 5 fold cross-validation :  0.4316094  mean rmse on test data :  3.949122 

```


<br>

#### **5 threads    ( high dimensional dataset and presence of multicollinearity )**



<br>

For the high-dimensional data (can be downloaded from my [Github repository](https://github.com/mlampros/DataSets)) I'll use the *FastRGF_Regressor* rather than the RGF_Regressor (comparison **without doing hyper-parameter tuning**),

<br>

```R

# download the data from my Github repository (tested on a Linux OS)

system("wget https://raw.githubusercontent.com/mlampros/DataSets/master/africa_soil_train_data.zip")


# load the data in the R session

train_dat = read.table(unz("africa_soil_train_data.zip", "train.csv"), nrows = 1157, header = T, quote = "\"", sep = ",")


# c("Ca", "P", "pH", "SOC", "Sand") : response variables            


# exclude response-variables and factor variable

x = train_dat[, -c(1, which(colnames(train_dat) %in% c("Ca", "P", "pH", "SOC", "Sand", "Depth")))]


# take (randomly) the first of the responses for train

y = train_dat[, "Ca"]


# dataset for ranger

tmp_rg_dat = cbind(Ca = y, x)


# cross-validation folds

set.seed(2)
FOLDS = regr_folds(folds = NUM_FOLDS, y, stratified = T)


highdim_rgf_te = highdim_ranger_te = highdim_xgb_te = highdim_rgf_time = highdim_ranger_time = highdim_xgb_time = rep(NA, NUM_FOLDS)


for (i in 1:length(FOLDS)) {

  cat("fold : ", i, "\n")

  new_samp = unlist(FOLDS[-i])
  new_samp_ = unlist(FOLDS[i])


  # RGF
  #----

  rgf_start = Sys.time()

  init_regr = FastRGF_Regressor$new(n_jobs = 5, l2 = 0.1)                  # I added 'l2' regularization

  init_regr$fit(x = as.matrix(x[new_samp, ]), y = y[new_samp])

  pr_te = init_regr$predict(as.matrix(x[new_samp_, ]))

  rgf_end = Sys.time()

  highdim_rgf_time[i] = rgf_end - rgf_start

  highdim_rgf_te[i] = MLmetrics::RMSE(y[new_samp_], pr_te)


  # ranger
  #-------

  ranger_start = Sys.time()
  

  fit = ranger(dependent.variable.name = "Ca", data = tmp_rg_dat[new_samp, ], 
               
               write.forest = TRUE, probability = F, num.threads = 5, num.trees = 500,
               
               verbose = T, classification = F, mtry = NULL, min.node.size = 5, 
               
               keep.inbag = T)
  

  pred_te = predict(fit, data = x[new_samp_, ], type = 'se')$predictions

  ranger_end = Sys.time()

  highdim_ranger_time[i] = ranger_end - ranger_start

  highdim_ranger_te[i] = MLmetrics::RMSE(y[new_samp_], pred_te)


  # xgboost
  #--------

  xgb_start = Sys.time()

  dtrain <- xgb.DMatrix(data = as.matrix(x[new_samp, ]), label = y[new_samp])

  dtest <- xgb.DMatrix(data = as.matrix(x[new_samp_, ]), label = y[new_samp_])

  watchlist <- list(train = dtrain, test = dtest)

  param = list("objective" = "reg:linear", "bst:eta" = 0.05, "max_depth" = 6, 
               
               "subsample" = 0.85, "colsample_bytree" = 0.85, "booster" = "gbtree",
               
               "nthread" = 5)                                                                     # "lambda" = 0.1 does not improve RMSE

  fit = xgb.train(param, dtrain, nround = 500, print_every_n  = 100, watchlist = watchlist,

                  early_stopping_rounds = 20, maximize = FALSE, verbose = 0)

  p_te = xgboost:::predict.xgb.Booster(fit, as.matrix(x[new_samp_, ]), ntreelimit = fit$best_iteration)

  xgb_end = Sys.time()

  highdim_xgb_time[i] = xgb_end - xgb_start

  highdim_xgb_te[i] = MLmetrics::RMSE(y[new_samp_], p_te)
}

```

<br>

```R

fold :  1 
fold :  2 
fold :  3 
fold :  4 
fold :  5 

```

<br>

```R

cat("total time rgf 5 fold cross-validation : ", sum(highdim_rgf_time), " mean rmse on test data : ", mean(highdim_rgf_te), "\n")

cat("total time ranger 5 fold cross-validation : ", sum(highdim_ranger_time), " mean rmse on test data : ", mean(highdim_ranger_te), "\n")

cat("total time xgb 5 fold cross-validation : ", sum(highdim_xgb_time), " mean rmse on test data : ", mean(highdim_xgb_te), "\n")

```

<br>

```R

total time rgf 5 fold cross-validation :  92.31971  mean rmse on test data :  0.5155166
total time ranger 5 fold cross-validation :  27.32866  mean rmse on test data :  0.5394164
total time xgb 5 fold cross-validation :  30.48834  mean rmse on test data :  0.5453544

```


<br>

The *README.md* file of the *RGF* package includes the SystemRequirements and installation instructions. 

An updated version of the RGF package can be found in my [Github repository](https://github.com/mlampros/RGF) and to report bugs/issues please use the following link, [https://github.com/mlampros/RGF/issues](https://github.com/mlampros/RGF/issues).


<br>
