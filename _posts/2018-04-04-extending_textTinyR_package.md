---
layout: post
title: New functionality for the textTinyR package
tags: [R, package]
comments: true
---


This blog post discuss the new functionality, which is added in the textTinyR package (version 1.1.0). I'll explain some of the functions by using the data and pre-processing steps of [this blog-post](https://miguelmalvarez.com/2015/03/20/classifying-reuters-21578-collection-with-python-representing-the-data/).

<br>

The following code chunks assume that the *nltk-corpus*  is already downloaded and the *reticulate* package is installed,

<br>

```R

NLTK = reticulate::import("nltk.corpus")

text_reuters = NLTK$reuters 


nltk = reticulate::import("nltk")

# if the 'reuters' data is not already available then it can be downloaded from within R

nltk$download('reuters')               
```

<br>

```R

documents = text_reuters$fileids()

str(documents)


# List of categories
categories = text_reuters$categories()

str(categories)


# Documents in a category
category_docs = text_reuters$fileids("acq")

str(category_docs)


one_doc = text_reuters$raw("test/14843")
one_doc

```

<br>

The collection originally consisted of 21,578 documents but a subset and split is traditionally used. The most common split is *Mod-Apte* which only considers 
categories that have at least one document in the training set and the test set. The *Mod-Apte* split has 90 categories with a training set of 7769 documents
and a test set of 3019 documents.

<br>

```R

documents = text_reuters$fileids()


# document ids for train - test
train_docs_id = documents[as.vector(sapply(documents, function(i) substr(i, 1, 5) == "train"))]
test_docs_id = documents[as.vector(sapply(documents, function(i) substr(i, 1, 4) == "test"))]


train_docs = lapply(1:length(train_docs_id), function(x) text_reuters$raw(train_docs_id[x]))
test_docs = lapply(1:length(test_docs_id), function(x) text_reuters$raw(test_docs_id[x]))

str(train_docs)
str(test_docs)


# train - test labels  [ some categories might have more than one label (overlapping) ]

train_labels = as.vector(sapply(train_docs_id, function(x) text_reuters$categories(x)))         
test_labels = as.vector(sapply(test_docs_id, function(x) text_reuters$categories(x)))  
```

<br>

### textTinyR - fastText - doc2vec - kmeans - cluster_medoids

<br>

First, I'll perform the following pre-processing steps : 

* convert to lower case
* trim tokens
* remove stopwords
* porter stemming 
* keep words with minimum number of characters equal to 3

<br>

```R
concat = c(unlist(train_docs), unlist(test_docs))
length(concat)


clust_vec = textTinyR::tokenize_transform_vec_docs(object = concat, as_token = T,
                                                   to_lower = T, 
                                                   remove_punctuation_vector = F,
                                                   remove_numbers = F, 
                                                   trim_token = T,
                                                   split_string = T,
                                                   split_separator = " \r\n\t.,;:()?!//", 
                                                   remove_stopwords = T,
                                                   language = "english", 
                                                   min_num_char = 3, 
                                                   max_num_char = 100,
                                                   stemmer = "porter2_stemmer", 
                                                   threads = 4,
                                                   verbose = T)

unq = unique(unlist(clust_vec$token, recursive = F))
length(unq)


# I'll build also the term matrix as I'll need the global-term-weights

utl = textTinyR::sparse_term_matrix$new(vector_data = concat, file_data = NULL,
                                        document_term_matrix = TRUE)

tm = utl$Term_Matrix(sort_terms = FALSE, to_lower = T, remove_punctuation_vector = F,
                     remove_numbers = F, trim_token = T, split_string = T, 
                     stemmer = "porter2_stemmer",
                     split_separator = " \r\n\t.,;:()?!//", remove_stopwords = T,
                     language = "english", min_num_char = 3, max_num_char = 100,
                     print_every_rows = 100000, normalize = NULL, tf_idf = F, 
                     threads = 6, verbose = T)

gl_term_w = utl$global_term_weights()
str(gl_term_w)
```

<br>

For simplicity, I'll use the *Reuters* data as input to the *fastText* algorithm. The data has to be first pre-processed and then saved to a file,

<br>

```R

 word_vecs_dir = '/path_to_your_folder/'            # the tail forward slash is required

 save_dat = textTinyR::tokenize_transform_vec_docs(object = concat, as_token = T, 
                                                   to_lower = T, 
                                                   remove_punctuation_vector = F,
                                                   remove_numbers = F, trim_token = T, 
                                                   split_string = T, 
                                                   split_separator = " \r\n\t.,;:()?!//",
                                                   remove_stopwords = T, language = "english", 
                                                   min_num_char = 3, max_num_char = 100, 
                                                   stemmer = "porter2_stemmer", 
                                                   path_2folder = word_vecs_dir, 
                                                   threads = 1,           # whenever I save data to file set the number threads to 1
                                                   verbose = T)
```

<br>

Then, I'll load the previously saved data and I'll use [fastText](https://github.com/mlampros/fastText) to build the word-vectors, 

<br>

```R

PATH_INPUT = glue::glue("{word_vecs_dir}output_token_single_file.txt")    # load the "output_token_single_file.txt" file
DIR_OUT = file.path(glue::glue("{word_vecs_dir}rt_fst_model"))            # directory where the .bin and .vec files will be saved
if (!dir.exists(DIR_OUT)) dir.create(DIR_OUT)

PATH_OUT = file.path(DIR_OUT, 'rt_fst_model')                             # file name that will take the '.bin' and '.vec' extensions

list_params = list(command = 'skipgram',
                   lr = 0.075,
                   dim = 300,
                   lrUpdateRate = 100,
                   ws = 5, 
                   epoch = 5,
                   minCount = 1, 
                   neg = 5,
                   wordNgrams = 2, 
                   loss = "ns", 
                   bucket = 2e+06,
                   minn = 0, 
                   maxn = 0, 
                   thread = 6,
                   t = 1e-04, 
                   verbose = 2,
                   input = PATH_INPUT,
                   output = PATH_OUT,
                   verbose = 2,
                   thread = 6)

vecs = fastText::fasttext_interface(list_params,
                                   path_output = file.path(DIR_OUT,"model_logs.txt"),
                                   MilliSecs = 100)
```

<br>

Before using one of the three methods, it would be better to reduce the initial dimensions of the word-vectors (rows of the matrix). So, I'll keep the word-vectors for which the terms appear in the *Reuters* data set - *clust_vec$token* ( although it's not applicable in this case, if the resulted word-vectors were based on external data - say the Wikipedia data - then their dimensions would be way larger and many of the terms would be redundant for the *Reuters* data set increasing that way the computation time considerably when invoking one of the doc2vec methods),

<br>

```R

init = textTinyR::Doc2Vec$new(token_list = clust_vec$token, 
                              word_vector_FILE = file.path(DIR_OUT, "rt_fst_model.vec"),
                              print_every_rows = 5000, 
                              verbose = TRUE, 
                              copy_data = FALSE)                  # use of external pointer


pre-processing of input data starts ...
File is successfully opened
total.number.lines.processed.input: 25000
creation of index starts ...
intersection of tokens and wordvec character strings starts ...
modification of indices starts ...
final processing of data starts ...
File is successfully opened
total.number.lines.processed.output: 25000
```

<br>

In case that *copy_data = TRUE* (in the previous "textTinyR::Doc2Vec$new()" function) then the user can observe the pre-processed data before using one of the 'doc2vec' methods (uncomment the next 2 lines if this is the case),

<br>

```R

# res_wv = init$pre_processed_wv()                           
# 
# str(res_wv)
```

<br>

Then, I can use one of the three methods (*sum_sqrt*, *min_max_norm*, *idf*) to receive the transformed vectors. These methods are based on the following *blog-posts* (see [here](https://www.linkedin.com/pulse/duplicate-quora-question-abhishek-thakur) and [here](http://www.erogol.com/duplicate-question-detection-deep-learning/) for references), 

<br>

```R

doc2_sum = init$doc2vec_methods(method = "sum_sqrt", threads = 6)
doc2_norm = init$doc2vec_methods(method = "min_max_norm", threads = 6)
doc2_idf = init$doc2vec_methods(method = "idf", global_term_weights = gl_term_w, threads = 6)

rows_cols = 1:5

doc2_sum[rows_cols, rows_cols]
doc2_norm[rows_cols, rows_cols]
doc2_idf[rows_cols, rows_cols]

> dim(doc2_sum)
[1] 10788   300
> dim(doc2_norm)
[1] 10788   300
> dim(doc2_idf)
[1] 10788   300
```

<br>

For illustration, I'll use the resulted word-vectors of the *sum_sqrt* method. The approach described can be used as an alternative to *Latent semantic indexing (LSI)* or *topic-modeling* in order to discover categories in text data (documents).

<br>

First, someone can seach for the optimal number of clusters using the *Optimal_Clusters_KMeans* function of the *ClusterR* package,

<br>


```R

scal_dat = ClusterR::center_scale(doc2_sum)     # center and scale the data


opt_cl = ClusterR::Optimal_Clusters_KMeans(scal_dat, max_clusters = 15, 
                                           criterion = "distortion_fK",
                                           fK_threshold = 0.85, num_init = 3, 
                                           max_iters = 50,
                                           initializer = "kmeans++", tol = 1e-04, 
                                           plot_clusters = TRUE,
                                           verbose = T, tol_optimal_init = 0.3, 
                                           seed = 1)

```

<br>

Based on the output of the *Optimal_Clusters_KMeans* function, I'll pick 5 as the optimal number of clusters in order to perform *k-means clustering*,

<br>

```R

num_clust = 5

km = ClusterR::KMeans_rcpp(scal_dat, clusters = num_clust, num_init = 3, max_iters = 50,
                           initializer = "kmeans++", fuzzy = T, verbose = F,
                           CENTROIDS = NULL, tol = 1e-04, tol_optimal_init = 0.3, seed = 2)


table(km$clusters)

   1    2    3    4    5 
 713 2439 2393 2607 2636 

```

<br>

As a follow up, someone can also perform *cluster-medoids* clustering using the *pearson-correlation* metric, which resembles the *cosine* distance ( the latter is frequently used for text clustering ),

<br>

```R

kmed = ClusterR::Cluster_Medoids(scal_dat, clusters = num_clust, 
                                 distance_metric = "pearson_correlation",
                                 minkowski_p = 1, threads = 6, swap_phase = TRUE, 
                                 fuzzy = FALSE, verbose = F, seed = 1)


table(kmed$clusters)

   1    2    3    4    5 
2396 2293 2680  875 2544 

```

<br>

Finally, the word-frequencies of the documents can be obtained using the *cluster_frequency* function, which groups the tokens (words) of the documents based on which cluster each document appears,

<br>

```R

freq_clust = textTinyR::cluster_frequency(tokenized_list_text = clust_vec$token, 
                                          cluster_vector = km$clusters, verbose = T)

Time difference of 0.1762383 secs

```

<br>

```R
> freq_clust

$`3`
         WORDS COUNTS
   1:      mln   8701
   2:      000   6741
   3:      cts   6260
   4:      net   5949
   5:     loss   4628
  ---                
6417:    vira>      1
6418:    gain>      1
6419:     pwj>      1
6420: drummond      1
6421: parisian      1

$`1`
         WORDS COUNTS
   1:      cts   1303
   2:   record    696
   3:    april    669
   4:      &lt    652
   5: dividend    554
  ---                
1833:     hvt>      1
1834:    bang>      1
1835:   replac      1
1836:    stbk>      1
1837:     bic>      1

$`4`
         WORDS COUNTS
    1:     mln   6137
    2:     pct   5084
    3:    dlrs   4024
    4:    year   3397
    5: billion   3390
   ---               
10968:   heijn      1
10969: "behind      1
10970:    myo>      1
10971:  "favor      1
10972: wonder>      1

$`5`
                  WORDS COUNTS
    1:              &lt   4244
    2:            share   3748
    3:             dlrs   3274
    4:          compani   3184
    5:              mln   2659
   ---                        
13059:        often-fat      1
13060: computerknowledg      1
13061:       fibrinolyt      1
13062:           hercul      1
13063:           ceroni      1

$`2`
             WORDS COUNTS
    1:       trade   3077
    2:        bank   2578
    3:      market   2535
    4:         pct   2416
    5:        rate   2308
   ---                   
13702:        "mfn      1
13703:         uk>      1
13704:    honolulu      1
13705:        arap      1
13706: infinitesim      1


```

<br>

```R

freq_clust_kmed = textTinyR::cluster_frequency(tokenized_list_text = clust_vec$token, 
                                               cluster_vector = kmed$clusters, verbose = T)

Time difference of 0.1685851 secs
```

<br>


This is one of the ways that the transformed word-vectors can be used and is solely based on tokens (words) and word frequencies. However a more advanced approach would be to cluster documents based on word *n-grams* and take advantage of *graphs* as explained [here](https://www.tidytextmining.com/ngrams.html#visualizing-bigrams-in-other-texts) in order to plot the nodes, edges and text.


<br><br>

*References*:

* https://miguelmalvarez.com/2015/03/20/classifying-reuters-21578-collection-with-python-representing-the-data/
* https://www.linkedin.com/pulse/duplicate-quora-question-abhishek-thakur
* http://www.erogol.com/duplicate-question-detection-deep-learning/
* https://www.tidytextmining.com/ngrams.html#visualizing-bigrams-in-other-texts


<br>

The package documentation includes more details for the new functions. The updated version of the *textTinyR* package can be found in my [Github repository](https://github.com/mlampros/textTinyR) and to report bugs/issues please use the following link, [https://github.com/mlampros/textTinyR/issues](https://github.com/mlampros/textTinyR/issues).

<br>
