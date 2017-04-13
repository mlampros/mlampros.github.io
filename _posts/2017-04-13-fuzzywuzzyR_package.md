---
layout: post
title: Fuzzy string Matching using fuzzywuzzyR and the reticulate package in R
tags: [R, package, R-bloggers]
comments: true
---


I recently released an (other one) R package on CRAN - **fuzzywuzzyR** - which ports the [fuzzywuzzy](https://github.com/seatgeek/fuzzywuzzy) python library in R. "fuzzywuzzy does fuzzy string matching by using the Levenshtein Distance to calculate the differences between sequences (of character strings)."
There is no *big news* here as in R already exist similar packages such as the [stringdist](https://github.com/markvanderloo/stringdist) package. Why then creating the package? Well, I intend to participate in a recently launched [kaggle competition](https://www.kaggle.com/c/quora-question-pairs) and one popular method to build features (predictors) is fuzzy string matching as explained in this [blog post](https://www.linkedin.com/pulse/duplicate-quora-question-abhishek-thakur). My (second) aim was to use the (newly released from Rstudio) [reticulate](https://github.com/rstudio/reticulate) package, which "provides an R interface to Python modules, classes, and functions" and makes the process of porting python code in R not cumbersome.
First, I'll explain the functionality of the **fuzzywuzzyR** package and then I'll give some examples on how to take advantage of the *reticulate* package in R.

<br>

#### **fuzzywuzzyR**

<br>

The *fuzzywuzzyR* package includes R6-classes / functions for string matching,

<br>

##### **classes** 


<br>


|    FuzzExtract              |   FuzzMatcher                   |  FuzzUtils                       |  SequenceMatcher        |
| :-------------------------: |  :----------------------------: | :-----------------------------:  | :---------------------: |
|   Extract()                 |  Partial_token_set_ratio()      | Full_process()                   | ratio()                 |
|   ExtractBests()            |  Partial_token_sort_ratio()     | Make_type_consistent()           | quick_ratio()           |
|   ExtractWithoutOrder()     |  Ratio()                        | Asciidammit()                    | real_quick_ratio()      |
|   ExtractOne()              |  QRATIO()                       | Asciionly()                      | get_matching_blocks()   |
|                             |  WRATIO()                       | Validate_string()                | get_opcodes()           |
|                             |  UWRATIO()                      |                                  |                         |
|                             |  UQRATIO()                      |                                  |                         |
|                             |  Token_sort_ratio()             |                                  |                         |
|                             |  Partial_ratio()                |                                  |                         |
|                             |  Token_set_ratio()              |                                  |                         |


<br>
  
  
##### **functions**


<br>

| GetCloseMatches() |
| :---------------- |

<br>


The following code chunks / examples are part of the package documentation and give an idea on what can be done with the *fuzzywuzzyR* package,

<br>


##### *FuzzExtract*

<br>

Each one of the methods in the *FuzzExtract* class takes a *character string* and a *character string sequence* as input ( except for the *Dedupe* method which takes a string sequence only ) and given a *processor* and a *scorer* it returns one or more string match(es) and the corresponding score ( in the range 0 - 100 ). Information about the additional parameters (*limit*, *score_cutoff* and *threshold*) can be found in the package documentation,

<br>

```R

library(fuzzywuzzyR)

word = "new york jets"

choices = c("Atlanta Falcons", "New York Jets", "New York Giants", "Dallas Cowboys")


#------------
# processor :
#------------

init_proc = FuzzUtils$new()      # initialization of FuzzUtils class to choose a processor

PROC = init_proc$Full_process    # processor-method

PROC1 = tolower                  # base R function ( as an example for a processor )

#---------
# scorer :
#---------

init_scor = FuzzMatcher$new()    # initialization of the scorer class

SCOR = init_scor$WRATIO          # choosen scorer function


init <- FuzzExtract$new()        # Initialization of the FuzzExtract class

init$Extract(string = word, sequence_strings = choices, processor = PROC, scorer = SCOR)
  
```
  

  
```R

# example output
  
  [[1]]
[[1]][[1]]
[1] "New York Jets"

[[1]][[2]]
[1] 100


[[2]]
[[2]][[1]]
[1] "New York Giants"

[[2]][[2]]
[1] 79


[[3]]
[[3]][[1]]
[1] "Atlanta Falcons"

[[3]][[2]]
[1] 29


[[4]]
[[4]][[1]]
[1] "Dallas Cowboys"

[[4]][[2]]
[1] 22
  
```

```R

# extracts best matches (limited to 2 matches)

init$ExtractBests(string = word, sequence_strings = choices, processor = PROC1,

                  scorer = SCOR, score_cutoff = 0L, limit = 2L)
                  
```

```R

[[1]]
[[1]][[1]]
[1] "New York Jets"

[[1]][[2]]
[1] 100


[[2]]
[[2]][[1]]
[1] "New York Giants"

[[2]][[2]]
[1] 79

```

```R

# extracts matches without keeping the output order

init$ExtractWithoutOrder(string = word, sequence_strings = choices, processor = PROC,

                         scorer = SCOR, score_cutoff = 0L)

```


```R

[[1]]
[[1]][[1]]
[1] "Atlanta Falcons"

[[1]][[2]]
[1] 29


[[2]]
[[2]][[1]]
[1] "New York Jets"

[[2]][[2]]
[1] 100


[[3]]
[[3]][[1]]
[1] "New York Giants"

[[3]][[2]]
[1] 79


[[4]]
[[4]][[1]]
[1] "Dallas Cowboys"

[[4]][[2]]
[1] 22

```


```R

# extracts first result 

init$ExtractOne(string = word, sequence_strings = choices, processor = PROC,

                scorer = SCOR, score_cutoff = 0L)

```


```R

[[1]]
[1] "New York Jets"

[[2]]
[1] 100

```
<br>

The *dedupe* method removes duplicates from a sequence of character strings using fuzzy string matching, 

<br>

```R

duplicat = c('Frodo Baggins', 'Tom Sawyer', 'Bilbo Baggin', 'Samuel L. Jackson',

             'F. Baggins', 'Frody Baggins', 'Bilbo Baggins')


init$Dedupe(contains_dupes = duplicat, threshold = 70L, scorer = SCOR)

```


```R

[1] "Frodo Baggins"     "Samuel L. Jackson" "Bilbo Baggins"     "Tom Sawyer"

```

<br>

##### *FuzzMatcher*

<br>

Each one of the methods in the *FuzzMatcher* class takes two *character strings* (string1, string2) as input and returns a score ( in range 0 to 100 ). Information about the additional parameters (*force_ascii*, *full_process* and *threshold*) can be found in the package documentation,

```R

s1 = "Atlanta Falcons"

s2 = "New York Jets"

init = FuzzMatcher$new()          initialization of FuzzMatcher class

init$Partial_token_set_ratio(string1 = s1, string2 = s2, force_ascii = TRUE, full_process = TRUE)

# example output

[1] 31

```
```R

init$Partial_token_sort_ratio(string1 = s1, string2 = s2, force_ascii = TRUE, full_process = TRUE)


[1] 31

```

```R

init$Ratio(string1 = s1, string2 = s2)

[1] 21

```

```R

init$QRATIO(string1 = s1, string2 = s2, force_ascii = TRUE)

[1] 29

```

```R

init$WRATIO(string1 = s1, string2 = s2, force_ascii = TRUE)

[1] 29

```

```R

init$UWRATIO(string1 = s1, string2 = s2)

[1] 29

```

```R

init$UQRATIO(string1 = s1, string2 = s2)

[1] 29

```

```R

init$Token_sort_ratio(string1 = s1, string2 = s2, force_ascii = TRUE, full_process = TRUE)

[1] 29

```

```R


init$Partial_ratio(string1 = s1, string2 = s2)

[1] 23

```

```R

init$Token_set_ratio(string1 = s1, string2 = s2, force_ascii = TRUE, full_process = TRUE)

[1] 29

```

<br>

##### *FuzzUtils*

<br>

The *FuzzUtils* class includes a number of utility methods, from which the *Full_process* method is from greater importance as besides its main functionality it can also be used as a secondary function in some of the other fuzzy matching classes,

<br>

```R

s1 = 'Frodo Baggins'

init = FuzzUtils$new()

init$Full_process(string = s1, force_ascii = TRUE)

```

```R

# example output

[1] "frodo baggins"

```

<br>

##### *GetCloseMatches*

<br>

The *GetCloseMatches* method returns a list of the best "good enough" matches. The parameter *string* is a sequence for which close matches are desired (typically a character string), and *sequence_strings* is a list of sequences against which to match the parameter *string* (typically a list of strings).

<br>

```R

vec = c('Frodo Baggins', 'Tom Sawyer', 'Bilbo Baggin')

str1 = 'Fra Bagg'

GetCloseMatches(string = str1, sequence_strings = vec, n = 2L, cutoff = 0.6)


```

```R

[1] "Frodo Baggins"

```


<br>

##### *SequenceMatcher*

<br>

The *SequenceMatcher* class is based on [difflib](https://www.npmjs.com/package/difflib) which comes by default installed with python and includes the following fuzzy string matching methods,

<br>


```R

s1 = ' It was a dark and stormy night. I was all alone sitting on a red chair.'

s2 = ' It was a murky and stormy night. I was all alone sitting on a crimson chair.'

init = SequenceMatcher$new(string1 = s1, string2 = s2)

init$ratio()

[1] 0.9127517

```

```R

init$quick_ratio()

[1] 0.9127517

```

```R

init$real_quick_ratio()

[1] 0.966443 

```
<br>

The *get_matching_blocks* and *get_opcodes* return triples and 5-tuples describing matching subsequences. More information can be found in the [Python's difflib module](https://www.npmjs.com/package/difflib) and in the *fuzzywuzzyR* package documentation.

<br>

A last think to note here is that the mentioned fuzzy string matching classes can be parallelized using the base R *parallel* package. For instance, the following *MCLAPPLY_RATIOS* function can take two vectors of character strings (QUERY1, QUERY2) and return the scores for each method of the *FuzzMatcher* class,

<br>

```R

MCLAPPLY_RATIOS = function(QUERY1, QUERY2, class_fuzz = 'FuzzMatcher', method_fuzz = 'QRATIO', threads = 1, ...) {

  init <- eval(parse(text = paste0(class_fuzz, '$new()')))

  METHOD = paste0('init$', method_fuzz)

  if (threads == 1) {

    res_qrat = lapply(1:length(QUERY1), function(x) do.call(eval(parse(text = METHOD)), list(QUERY1[[x]], QUERY2[[x]], ...)))}

  else {

    res_qrat = parallel::mclapply(1:length(QUERY1), function(x) do.call(eval(parse(text = METHOD)), list(QUERY1[[x]], QUERY2[[x]], ...)), mc.cores = threads)
  }

  return(res_qrat)
}

```
<br>

```R

query1 = c('word1', 'word2', 'word3')

query2 = c('similarword1', 'similar_word2', 'similarwor')

quer_res = MCLAPPLY_RATIOS(query1, query2, class_fuzz = 'FuzzMatcher', method_fuzz = 'QRATIO', threads = 1)

unlist(quer_res)

```

```R

# example output

[1] 59 56 40

```


<br><br>

#### **reticulate** package

<br>

My personal opinion is that the newly released [reticulate](https://github.com/rstudio/reticulate) package is *good news* (for all R-users with minimal knowledge of python) and *bad news* (for package maintainers whose packages do not cover the full spectrum of a subject in comparison to an existing python library) at the same time. I'll explain this in the following two examples.

<br>

As an R user I'd always liked to have a *truncated svd* function similar to the one of the [sklearn](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html) python library. So, now in R using the reticulate package and the [mnist data](https://github.com/mlampros/DataSets) set one can do,

<br>

```R

reticulate::py_module_available('sklearn')       # check that 'sklearn' is available in your OS

[1] TRUE

```


```R

dim(mnist)                # after downloading and open the data from the previous link

70000   785

```

```R

mnist = as.matrix(mnist)                                  # convert to matrix

trunc_SVD = reticulate::import('sklearn.decomposition')

res_svd = trunc_SVD$TruncatedSVD(n_components = 100L, n_iter = 5L, random_state = 1L)

res_svd$fit(mnist)

# TruncatedSVD(algorithm='randomized', n_components=100, n_iter=5,
#       random_state=1, tol=0.0)
       
```

```R

out_svd = res_svd$transform(mnist)

str(out_svd)

# num [1:70000, 1:100] 1752 1908 2289 2237 2236 ...

```

```R

class(out_svd)

# [1] "matrix"

```
<br>

to receive the desired output ( a matrix with 70000 rows and 100 columns (components) ).

<br>

As a package maintainer, I do receive from time to time e-mails from users of my packages. In one of them a user asked me if the hog function of the [OpenImageR](https://github.com/mlampros/OpenImageR) package is capable of plotting the hog features. Actually not, but now an R-user can, for instance, use the [scikit-image](http://scikit-image.org/) python library to [plot the hog-features](http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html) using the following code chunk,

<br>

```R

reticulate::py_module_available("skimage")             # check that 'sklearn' is available in your OS

# [1] TRUE

```


```R

feat <- reticulate::import("skimage.feature")        # import module

data_sk <- reticulate::import("skimage.data")        # import data

color <- reticulate::import("skimage.color")         # import module to plot    

tmp_im = data_sk$astronaut()                         # import specific image data ('astronaut')

dim(tmp_im)

# [1] 512 512   3

```

```R

image = color$rgb2gray(tmp_im)                       # convert to gray
dim(image)

# [1] 512 512

```

```R

res = feat$hog(image, orientations = 8L, pixels_per_cell = c(16L, 16L), cells_per_block = c(1L, 1L), visualise=T)
str(res)

# List of 2
#  $ : num [1:8192(1d)] 1.34e-04 1.53e-04 6.68e-05 9.19e-05 7.93e-05 ...
#  $ : num [1:512, 1:512] 0 0 0 0 0 0 0 0 0 0 ...

```

```R 

OpenImageR::imageShow(res[[2]])       # using the OpenImageR to plot the data


```

![](hog_plot_astronaut.png)

<br>

As a final word, I think that the *reticulate* package, although not that popular yet, it will make a difference in the R-community.

<br>

The *README.md* file of the *fuzzywuzzyR* package includes the SystemRequirements and detailed installation instructions for each OS. 

An updated version of the fuzzywuzzyR package can be found in my [Github repository](https://github.com/mlampros/fuzzywuzzyR) and to report bugs/issues please use the following link, [https://github.com/mlampros/fuzzywuzzyR/issues](https://github.com/mlampros/fuzzywuzzyR/issues).


<br>
