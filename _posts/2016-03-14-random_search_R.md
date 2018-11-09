---
layout: post
title: Random search and resampling techniques in R 
tags: [R]
comments: true
---

This blog post is about randomly searching for the optimal parameters of various algorithms employing resampling in R. A [**randomized search**](https://en.wikipedia.org/wiki/Hyperparameter_optimization) simply samples parameter settings a fixed number of times from a specified subset of the hyperparameter space of a learning algorithm. This method has been found to be more effective in high-dimensional spaces than an exhaustive search (grid-search). Moreover, the purpose of random search is to optimize the performance of an algorithm using a **resampling method** such as cross-validation, bootstrapping etc. for a better generalization. 
<br>
<br>

### RandomSearchR

For the purpose of this post, I created the package **RandomSearchR**, which returns the optimal parameters for a variety of R models. The package can be installed from [Github](https://github.com/mlampros/RandomSearchR) using the install_github('mlampros/RandomSearchR') function of the devtools package (required is the most recent release of devtools). I'll employ both regression and classification data sets to illustrate the results of the RandomSearchR package functions.
<br>
<br>

### Random search for regression
<br>
For regression I'll utilize the Boston data set,

```R
library(MASS)
data(Boston)

X = Boston[, -dim(Boston)[2]]
y1 = Boston[, dim(Boston)[2]]
y_elm = matrix(Boston[, ncol(Boston)], nrow = length(Boston[, ncol(Boston)]), ncol = 1)

form <- as.formula(paste('medv ~', paste(names(X), collapse = '+')))  

ALL_DATA = Boston
```
<br>


The main function of the *RandomSearchR* package is the **random_search_resample** function, which takes the following arguments as shown in the case of the extreme learning machines

```R
# extreme learning machines

grid_extrLm = list(nhid = 5:50, actfun = c('sig', 'sin', 'purelin', 'radbas', 'relu'))


res_exttLm = random_search_resample(y = y1, tune_iters = 30,
                             
                                    resampling_method = list(method = 'cross_validation', repeats = NULL, sample_rate = NULL, folds = 5),
                             
                                    ALGORITHM = list(package = require(elmNNRcpp), algorithm = elm_train), 
                             
                                    grid_params = grid_extrLm, 
                                   
                                    DATA = list(x = as.matrix(X), y = y_elm),
                                   
                                    Args = NULL,
                                   
                                    regression = TRUE, re_run_params = FALSE)
```
<br>



* **y** is the response variable
* **tune_iters** is the number of times the algorithm should be run. In each iteration, a different subset of parameters from the grid_extLm list will be fitted to the algorithm.
* **resampling_method**,  there are 3 methods available: cross-validation, bootstrap and train-test-split. If more than one algorithm will be employed, then it's recommended to use the same resampling technique, so that a performance comparison between models is possible
* **ALGORITHM**, takes the package and the function to be used as parameters, here the *elmtrain* function of the *elmNNRcpp* package
* **grid_params** takes the defined grid of parameters, here the *nhid* and *actfun* of the *elmtrain* function
* **DATA** should be a list with the data. The following forms, as they appear in most of the packages, can be included: (x, y), (formula, data) or (target, data). In order to make xgboost and h2o work, I made some modifications, thus for **xgboost** the DATA should be a *watchlist*, such as **watchlist = list(label = y1, data = X)** and for **h2o** a pair of the following form (h2o.x, h2o.y), i.e. **list(h2o.x = X, h2o.y = y1)**
* **Args** list takes arguments that are necessary for the function, such as *scale = TRUE* (it can be also NULL, to indicate that no further arguments for the function are needed)
* **regression** should be TRUE in regression tasks so that predictions are returned in the correct form.
* **re_run_params** should be TRUE only in case that the optimized parameters should be re-run for performance comparison with other models.
<br>

The *res_exttLm* object returns 2 lists, the *PREDS*, which includes the train-predictions (pred_tr), the test-predictions (pred_te), the response values for train (y_tr), and the response values for test (y_te) of each sample. Furthermore it returns the random parameters of each iteration *PARAMS*,

```R
# > str(res)
# List of 2
#  $ PREDS :List of 30
#   ..$ :List of 5
#   .. ..$ :List of 4
#   .. .. ..$ pred_tr: num [1:404] 27.9 27.9 25.6 23.6 23.6 ...
#   .. .. ..$ pred_te: num [1:102] 26.8 27.9 23.6 21.3 18.5 ...
#   .. .. ..$ y_tr   : num [1:404] 34.7 36.2 22.9 16.5 18.9 21.7 20.2 13.1 13.5 24.7 ...
#   .. .. ..$ y_te   : num [1:102] 24 21.6 15 13.9 13.2 21 30.8 20 19.4 16 ...
#   .. ..$ :List of 4
#   .. .. ..$ pred_tr: num [1:405] 25.5 25.9 24.9 13.7 10 ...
#   .. .. ..$ pred_te: num [1:101] 25.9 27 25.5 24.2 25.5 ...
#   .. .. ..$ y_tr   : num [1:405] 24 21.6 15 13.9 13.2 21 30.8 20 19.4 16 ...
#   .. .. ..$ y_te   : num [1:101] 34.7 36.2 22.9 16.5 18.9 21.7 20.2 13.1 13.5 24.7 ...
#   ....
#
#  $ PARAMS:'data.frame':	30 obs. of  2 variables:
#   ..$ nhid  : chr [1:30] "50" "40" "16" "30" ...
#   ..$ actfun: chr [1:30] "sig" "sig" "purelin" "sig" ...
#   ....
```

<br>
The following algorithms were tested and can be run in regression and classification error-free, 

|    package      |   algorithm        |  regression   |  binary classification   |   multiclass classification   |
| :-----------:   |  :-------------:   | :-----------: | :----------------------: | :---------------------------: |  
|   elmNNRcpp     |  elm_train         |    **x**      |             **x**        |              **x**            | 
|   randomForest  |  randomForest      |    **x**      |             **x**        |              **x**            | 
|   kernlab       |  ksvm              |    **x**      |             **x**        |              **x**            |
|   caret         |  knnreg, knn3      |    **x**      |             **x**        |              **x**            |
|   RWeka         |  IBk               |    **x**      |             **x**        |              **x**            |
|   RWeka         |  AdaBoostM1        |               |             **x**        |              **x**            |
|   RWeka         |  Bagging           |    **x**      |             **x**        |              **x**            |
|   RWeka         |  LogitBoost        |               |             **x**        |              **x**            |
|   RWeka         |  J48               |               |             **x**        |              **x**            |
|   RWeka         |  M5P               |    **x**      |                          |                               |
|   RWeka         |  M5Rules           |    **x**      |                          |                               |
|   RWeka         |  SMO               |               |             **x**        |              **x**            |
|   gbm           |  gbm               |    **x**      |             **x**        |              **x**            |
|   h2o           |  h2o.randomForest  |    **x**      |             **x**        |              **x**            |
|   h2o           |  h2o.deeplearning  |    **x**      |             **x**        |              **x**            |
|   h2o           |  h2o.gbm           |    **x**      |             **x**        |              **x**            |
|   xgboost       | xgb.train          |    **x**      |             **x**        |              **x**            |
|   e1071         | svm                |    **x**      |             **x**        |              **x**            |
|   LiblineaR     | LiblineaR          |    **x**      |             **x**        |              **x**            |
|   extraTrees    | extraTrees         |    **x**      |             **x**        |              **x**            |
|   glmnet        | cv.glmnet          |    **x**      |             **x**        |              **x**            |
|   nnet          |  nnet              |    **x**      |             **x**        |              **x**            |
|   ranger        |  ranger            |    **x**      |             **x**        |              **x**            |



**Worth mentioning**

* it is a good practice (in case of the RandomSearchR package) to always use **library(RandomSearchR)** before utilizing any of the functions.
* detailed examples with code of the previous table can be found on the [Github repository](https://github.com/mlampros/RandomSearchR/blob/master/tests/testthat/test-randomsearch.R)
* depending on the number of optimized parameters some algorithms can be run fewer/more times than other (for instance knn will be run fewer times than svm)
* to hide the progressbar when xgboost runs adjust the 'verbose' argument (set it to 0). 
If it happens that you use the package and came across a problem then commit an issue on github.
<br>
<br>
To continue here are some other grid-examples for illustration,

```R
# random forest

grid_rf = list(ntree = seq(30, 50, 5), mtry = c(2:3), nodesize = seq(5, 15, 5))


res_rf = random_search_resample(y1, tune_iters = 30, 
                              
                              resampling_method = list(method = 'cross_validation', repeats = NULL, sample_rate = NULL, folds = 5),
                              
                              ALGORITHM = list(package = require(randomForest), algorithm = randomForest), 
                              
                              grid_params = grid_rf, 
                              
                              DATA = list(x = X, y = y1),
                              
                              Args = NULL,
                              
                              regression = TRUE, re_run_params = FALSE)
```
<br>

```R
# xgboost

grid_xgb = list(params = list("objective" = "reg:linear", "bst:eta" = seq(0.05, 0.1, 0.005), "subsample" = seq(0.65, 0.85, 0.05), 

                              "max_depth" = seq(3, 5, 1), "eval_metric" = "rmse", "colsample_bytree" = seq(0.65, 0.85, 0.05), 
                              
                              "lambda" = 1e-5, "alpha" = 1e-5, "nthread" = 4))


res_xgb = random_search_resample(y1, tune_iters = 30, 
                              
                              resampling_method = list(method = 'cross_validation', repeats = NULL, sample_rate = NULL, folds = 5),
                              
                              ALGORITHM = list(package = library(xgboost), algorithm = xgb.train), 
                              
                              grid_params = grid_xgb, 
                              
                              DATA = list(watchlist = list(label = y1, data = X)),
                              
                              Args = list(nrounds = 200, verbose = 0, print_every_n = 500, early_stop_round = 30, maximize = FALSE),
                              
                              regression = TRUE, re_run_params = FALSE)
```
<br>

```R
# h2o gbm

grid_h2o = list(ntrees = seq(30, 50, 5), max_depth = seq(3, 5, 1), min_rows = seq(5, 15, 1), learn_rate = seq(0.01, 0.1, 0.005))


res_h2o = capture.output(random_search_resample(y1, tune_iters = 30, 
                               
                               resampling_method = list(method = 'cross_validation', repeats = NULL, sample_rate = NULL, folds = 5),
                               
                               ALGORITHM = list(package = require(h2o), algorithm = h2o.gbm), 
                               
                               grid_params = grid_h2o, 
                               
                               DATA = list(h2o.x = X, h2o.y = y1),
                               
                               Args = list(distribution = "gaussian", stopping_metric = "AUTO"),
                               
                               regression = TRUE, re_run_params = FALSE, max_mem_size = '4g', nthreads = 6))
```

<br>
<br>

### Random search for classification
<br>

The same function **random_search_resample** can be applied to classification tasks, 

```R
library(kknn)
data(glass)

X = glass[, -c(1, dim(glass)[2])]        # remove index and response
y1 = glass[, dim(glass)[2]]

form <- as.formula(paste('Type ~', paste(names(X),collapse = '+')))

y1 = c(1:length(unique(y1)))[ match(y1, sort(unique(y1))) ]             # labels should begin from 1:Inf

ALL_DATA = glass
ALL_DATA$Type = as.factor(y1)
```
<br>

Important here is that the labels are factors and begin from 1, this modification was necessary so that all package functions could work. Moreover, the *regression* argument should be set to FALSE. Here are some example grids for classification using *bootstrap* as a resampling method,

```R
# Extra Trees classifier

grid_extTr = list(ntree = seq(30, 50, 5), mtry = c(2:3), nodesize = seq(5, 15, 5))


res_extT = random_search_resample(as.factor(y1), tune_iters = 30, 
                              
                              resampling_method = list(method = 'bootstrap', repeats = 25, sample_rate = 0.65, folds = NULL),
                              
                              ALGORITHM = list(package = require(extraTrees), algorithm = extraTrees), 
                              
                              grid_params = grid_extTr, 
                              
                              DATA = list(x = X, y = as.factor(y1)),
                              
                              Args = NULL,
                              
                              regression = FALSE, re_run_params = FALSE)
```
<br>

```R
# kernel knn

grid_kknn = list(k = 3:20, distance = c(1, 2), kernel = c("rectangular", "triangular", "epanechnikov", "biweight", "triweight", 

                                                          "cos", "inv", "gaussian", "rank", "optimal"))


res_kkn = random_search_resample(as.factor(y1), tune_iters = 30, 
                              
                              resampling_method = list(method = 'bootstrap', repeats = 25, sample_rate = 0.65, folds = NULL),
                              
                              ALGORITHM = list(package = require(kknn), algorithm = kknn), 
                              
                              grid_params = grid_kknn, 
                              
                              DATA = list(formula = form, train = ALL_DATA),
                              
                              Args = NULL,
                              
                              regression = FALSE, re_run_params = FALSE)
```
<br>

```R
# support vector classifier

grid_kern = list(type = c('C-svc', 'C-bsvc'), C = c(0.1, 0.5, 1, 2, 10, 100), nu = c(0.1, 0.2, 0.5))


res_kern = random_search_resample(as.factor(y1), tune_iters = 30, 
                               
                              resampling_method = list(method = 'bootstrap', repeats = 25, sample_rate = 0.65, folds = NULL),
                              
                              ALGORITHM = list(package = require(kernlab), algorithm = ksvm), 
                              
                              grid_params = grid_kern, 
                              
                              DATA = list(x = as.matrix(X), y = as.factor(y1)),
                              
                              Args = list(prob.model = TRUE, scaled = FALSE),
                              
                              regression = FALSE, re_run_params = FALSE)
```
<br>

```R
# RWeka J48
#                    [ RWeka::WOW("J48") : gives info for the parameters of the RWeka control list ]

j48_lst = list(control = RWeka::Weka_control(B = c(TRUE, FALSE), M = seq(5, 15, 5), A = c(TRUE, FALSE)), R = c(TRUE, FALSE))


res_j48 = random_search_resample(as.factor(y1), tune_iters = 30, 

                               resampling_method = list(method = 'bootstrap', repeats = 25, sample_rate = 0.65, folds = NULL),
                               
                               ALGORITHM = list(package = require(RWeka), algorithm = J48), 
                               
                               grid_params = j48_lst, 
                               
                               DATA = list(formula = form, data = ALL_DATA),
                               
                               Args = NULL,
                               
                               regression = FALSE, re_run_params = FALSE)
```
<br>
<br>

### Model selection

Assuming that multiple model algorithms have been fitted, as previously in classification examples, then it would be of interest to observe which grid parameters optimize the evaluation metric. This can be done using the **performance_measures** function, which takes as arguments : *a list of the resulted objects*, *the evaluation metric* and *the sorting method of the results*. In the *performance measures* function we'll try to maximize the evaluation metric (here accuracy), 

```R
acc = function(y_true, preds) {             
  
  out = table(y_true, max.col(preds, ties.method = "random"))
  
  acc = sum(diag(out))/sum(out)
  
  acc
}
```

The evaluation metric can be customized to any other metric as long as the order of the arguments is (y_true, preds).
<br>
<br>
<br>
The next example function uses the four algorithm-grids illustrated in classification,

```R
perf = performance_measures(list_objects = list(extT = res_extT, kknn = res_kkn, ksvm = res_kern, j48_weka = res_j48), 

                            eval_metric = acc, 
                            
                            sort = list(variable = 'Median', decreasing = TRUE))

perf
```
<br>

This function returns four values: the first is a list of the grid parameters evaluated on the train data, the second is the same list of parameters evaluated on test data, the third gives summary statistics for the predictions of each algorithm compared with the other algorithms and the fourth list shows if any of the models had missing values in the predictions. The following is example output of the results,

```R
# 1st list TRAIN DATA

# extra trees

#    ntree mtry nodesize   Min. 1st Qu. Median   Mean 3rd Qu.   Max.
# 6     50    3        5 0.9825  0.9825 0.9826 0.9848  0.9883 0.9883
# 13    35    3        5 0.9766  0.9766 0.9825 0.9836  0.9884 0.9942
# 7     50    3        5 0.9766  0.9766 0.9767 0.9790  0.9825 0.9825
# 10    50    2        5 0.9649  0.9766 0.9766 0.9790  0.9883 0.9884
# 24    40    2        5 0.9532  0.9708 0.9766 0.9720  0.9767 0.9825
# 4     35    2        5 0.9591  0.9591 0.9709 0.9685  0.9766 0.9766
# 26    30    2        5 0.9649  0.9708 0.9708 0.9731  0.9766 0.9826

# ksvm

#      type   C  nu   Min. 1st Qu. Median   Mean 3rd Qu.   Max.
# 5   C-svc 0.1 0.2 0.5146  0.5731 0.7151 0.6705  0.7719 0.7778
# 15 C-bsvc 100 0.1 0.6550  0.6608 0.6784 0.6705  0.6784 0.6802
# 16  C-svc 100 0.1 0.6608  0.6608 0.6784 0.6717  0.6784 0.6802
# 4  C-bsvc 100 0.1 0.6667  0.6725 0.6725 0.6741  0.6784 0.6802
# 26 C-bsvc 100 0.1 0.6667  0.6667 0.6725 0.6729  0.6744 0.6842
# 10 C-bsvc   1 0.1 0.5965  0.6257 0.6608 0.6776  0.6919 0.8129
# 29  C-svc  10 0.5 0.6257  0.6395 0.6433 0.6437  0.6491 0.6608
# 28 C-bsvc  10 0.5 0.6199  0.6199 0.6374 0.6448  0.6570 0.6901

# ....

# 2nd list TEST DATA

# kknn

#     k distance       kernel   Min. 1st Qu. Median   Mean 3rd Qu.   Max.
# 3   6        2          cos 0.5814  0.6279 0.6744 0.6545  0.6744 0.7143
# 5  13        1 epanechnikov 0.5581  0.6047 0.6744 0.6591  0.7143 0.7442
# 6   6        2      optimal 0.5952  0.6279 0.6744 0.6586  0.6744 0.7209
# 14 15        1     biweight 0.6512  0.6512 0.6744 0.6824  0.7143 0.7209
# 18  5        2    triweight 0.6047  0.6279 0.6667 0.6729  0.7209 0.7442
# 11 12        1    triweight 0.5581  0.6047 0.6512 0.6501  0.6744 0.7619
# 12  7        1          cos 0.5814  0.6279 0.6512 0.6499  0.6512 0.7381
# 15  7        2      optimal 0.5952  0.6279 0.6512 0.6539  0.6744 0.7209


# j48

#        B  M     A   Min. 1st Qu. Median   Mean 3rd Qu.   Max.
# 2   TRUE 10 FALSE 0.5581  0.6279 0.6512 0.6780  0.7619 0.7907
# 8  FALSE 10  TRUE 0.5581  0.6279 0.6512 0.6780  0.7619 0.7907
# 15  TRUE 10 FALSE 0.5581  0.6279 0.6512 0.6780  0.7619 0.7907
# 20  TRUE 10 FALSE 0.5581  0.6279 0.6512 0.6780  0.7619 0.7907
# 23  TRUE 10 FALSE 0.5581  0.6279 0.6512 0.6780  0.7619 0.7907
# 1   TRUE  5  TRUE 0.3488  0.4762 0.5581 0.5138  0.5814 0.6047
# 5  FALSE  5  TRUE 0.3488  0.4762 0.5581 0.5138  0.5814 0.6047
# 9   TRUE  5 FALSE 0.3488  0.4762 0.5581 0.5138  0.5814 0.6047

# ....

# 3rd list resampling of the models

# $train_resampling
#                Min.   1st Qu.    Median      Mean   3rd Qu.      Max.
# kknn_regr 0.9027867 0.9092467 0.9189067 0.9162700 0.9221067 0.9283433
# extT      0.8137300 0.8497600 0.8634567 0.8692200 0.9003133 0.9188367
# ksvm      0.5936567 0.6070500 0.6203667 0.6286933 0.6369667 0.6854200
# j48_weka  0.5992267 0.7249800 0.7471433 0.7293533 0.7636433 0.8118833
# 
# $test_resampling
#                Min.   1st Qu.    Median      Mean   3rd Qu.      Max.
# kknn_regr 0.5406333 0.5900000 0.6143467 0.6167300 0.6462900 0.6924100
# extT      0.4100633 0.4732967 0.5249333 0.5142867 0.5607867 0.6023967
# ksvm      0.3968733 0.4224800 0.4589233 0.4675633 0.4943067 0.5652100
# j48_weka  0.3976333 0.4943433 0.5457167 0.5307600 0.5975333 0.6186267

# 4th list NA's in predictions

# $NAs_in_predictions
# kknn_regr      extT      ksvm  j48_weka 
#         0         0         0         0 
```
<br>

Definitely, not all grid-parameters maximized the evaluation metric, so we can keep a subset of them to re-run them on the same resampling method. Subsets can be taken using the **subset_mods** function, which takes 3 arguments:
the result from the *performance_measures* function, the number of best-performing grid-models (here 5) and a boolean, meaning, if the train or test predictions should be taken into account when subsetting the data.frames. It returns a list with the optimal parameters for each algorithm,

```R
bst_m = subset_mods(perf_meas_OBJ = perf, bst_mods = 5, train_params = FALSE)

bst_m


# $kknn
# $kknn$k
# [1]  6 13  6 15  5
# 
# $kknn$distance
# [1] 2 1 2 1 2
# 
# $kknn$kernel
# [1] "cos"          "epanechnikov" "optimal"      "biweight"     "triweight"   
# 
# 
# $extT
# $extT$ntree
# [1] 40 50 30 45 45
# 
# $extT$mtry
# [1] 2 3 2 2 3
# 
# $extT$nodesize
# [1]  5  5 10 15 10
# 
# 
# $ksvm
# $ksvm$type
# [1] "C-svc" "C-svc" "C-svc" "C-svc" "C-svc"
# 
# $ksvm$C
# [1] 2 2 1 2 2
# 
# $ksvm$nu
# [1] 0.1 0.2 0.5 0.2 0.5
# 
# 
# $j48_weka
# $j48_weka$B
# [1] "TRUE"  "FALSE" "TRUE"  "TRUE"  "TRUE" 
# 
# $j48_weka$M
# [1] 10 10 10 10 10
# 
# $j48_weka$A
# [1] "FALSE" "TRUE"  "FALSE" "FALSE" "FALSE"
```
<br>

Using the optimal parameters we can re-run the algorithms, however the *tune_iters* argument is adjusted to the number of optimal grid-params which is 5, thus any given number won't be taken into consideration. Furthermore the *grid_params* for each algorithm is the list of optimal parameters resulted of the previous object *bst_m* and the *re_run_params* should be set to TRUE,

```R
res2_extTr = random_search_resample(as.factor(y1), tune_iters = 30, 
                              
                              resampling_method = list(method = 'cross_validation', repeats = NULL, sample_rate = NULL, folds = 5),
                              
                              ALGORITHM = list(package = require(extraTrees), algorithm = extraTrees), 
                              
                              grid_params = bst_m$extT, 
                              
                              DATA = list(x = X, y = as.factor(y1)),
                              
                              Args = NULL,
                              
                              regression = FALSE, re_run_params = TRUE)
```
<br>

```R
res2_kknn = random_search_resample(as.factor(y1), tune_iters = 30, 

                              resampling_method = list(method = 'cross_validation', repeats = NULL, sample_rate = NULL, folds = 5),
                              
                              ALGORITHM = list(package = require(kknn), algorithm = kknn), 
                              
                              grid_params = bst_m$kknn, 
                              
                              DATA = list(formula = form, train = ALL_DATA),
                              
                              Args = NULL,
                              
                              regression = FALSE, re_run_params = TRUE)
```
<br>

```R
res2_ksvm = random_search_resample(as.factor(y1), tune_iters = 30, 
                               
                               resampling_method = list(method = 'cross_validation', repeats = NULL, sample_rate = NULL, folds = 5),
                               
                               ALGORITHM = list(package = require(kernlab), algorithm = ksvm), 
                               
                               grid_params = bst_m$ksvm, 
                               
                               DATA = list(x = as.matrix(X), y = as.factor(y1)),
                               
                               Args = list(prob.model = TRUE, scaled = FALSE),
                               
                               regression = FALSE, re_run_params = TRUE)
```
<br>

```R
res2_j48 = random_search_resample(as.factor(y1), tune_iters = 30, 
                               
                               resampling_method = list(method = 'cross_validation', repeats = NULL, sample_rate = NULL, folds = 5),
                               
                               ALGORITHM = list(package = require(RWeka), algorithm = J48), 
                               
                               grid_params = bst_m$j48_weka, 
                               
                               DATA = list(formula = form, data = ALL_DATA),
                               
                               Args = NULL,
                               
                               regression = FALSE, re_run_params = TRUE)
```
<br>

Now, using the optimized parameters for each model on the same folds we can run the **model_selection** function, which returns the t.test statistic, correlation and evaluation metric (here accuracy) for each pair of the selected algorithms. The function takes seven arguments : a list of the algorithms (which were re-run previously), a boolean if the statistics should be run on train or test predictions, an evaluation metric in form of a string (here 'acc'), the t.test confidence interval (defaults to 0.95), the correlation test (one of spearman, pearson, kendal) and a boolean if the evaluation metric should be sorted in decreasing order,

```R
tmp_lst = list(extT = res2_extTr, kknn = res2_kknn, ksvm = res2_ksvm, j48_weka = res2_j48)


res = model_selection(tmp_lst, on_Train = FALSE, regression = FALSE, 

                     evaluation_metric = 'acc', t.test.conf.int = 0.95, 
                     
                     cor_test = list(method = 'spearman'), sort_decreasing = TRUE)

res


#   algorithm_1 algorithm_2  ||  t.test.p.value t.test.conf.int.min t.test.conf.int.max t.test.mean.of.diffs  ||
# 1        kknn        extT  ||          0.0000              0.0885              0.1618               0.1252  ||
# 2        kknn        ksvm  ||          0.0000              0.1530              0.2256               0.1893  ||
# 3        kknn    j48_weka  ||          0.4332             -0.0448              0.0198              -0.0125  ||
# 4        extT        ksvm  ||          0.0034              0.0234              0.1049               0.0641  ||
# 5        extT    j48_weka  ||          0.0000             -0.1861             -0.0892              -0.1376  ||
# 6        ksvm    j48_weka  ||          0.0000             -0.2471             -0.1564              -0.2017  ||


#                            ||  spearman_estimate.rho spearman_p.value  ||  acc_algorithm_1 acc_algorithm_2
#                            ||                 0.7873           0.0001  ||        0.6654928       0.5403322
#                            ||                 0.6631           0.0101  ||        0.6654928       0.4762126
#                            ||                 0.7129           0.0070  ||        0.6654928       0.6779623
#                            ||                 0.8011           0.0000  ||        0.5403322       0.4762126
#                            ||                 0.7753           0.0000  ||        0.5403322       0.6779623
#                            ||                 0.7227           0.0000  ||        0.4762126       0.6779623
```

<br>

The t.test, generally, can be used to determine if two sets of data are significantly different from each other, under the assumption that the differences between the values are normally distributed. So, under the assumption of normality one can reject the null hypothesis that there is no difference between two algorithmic models, if the t.test.p.value is lower than 0.05 (significance level 5%). The correlation test, similarly, shows if a correlation is present between the algorithms using also a p.value for assessment.
