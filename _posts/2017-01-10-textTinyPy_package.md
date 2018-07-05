---
layout: post
title: Text Processing using the textTinyPy package in Python
tags: [python, package]
comments: true
---




This blog post (which has many similarities with the [previous one](http://mlampros.github.io/2017/01/05/textTinyR_package/)) explains the functionality of the **textTinyPy** package which can be installed from [**pypi**](https://pypi.python.org/pypi/textTinyPy/0.0.1/) using,

* **pip install textTinyPy**

The package has been tested on Linux using python 2.7. It is based on the same C++ source code as the [*textTinyR*](https://github.com/mlampros/textTinyR) package, but it has a slightly different structure and it's wrapped in Python using Cython. It will work properly only if the following requirements are satisfied / installed: <br>

<br>

#### **System Requirements:**

<br>

* **boost** [(boost >= 1.55)](http://www.boost.org/)
* **armadillo** [(armadillo >= 0.7.5)](http://arma.sourceforge.net/)
* a **C++11** compiler
* [OpenMP](http://www.openmp.org/) for parallelization (optional)

<br>

#### **Python Requirements:**

<br>

* **Cython**>=0.23.5
* **pandas**>=0.13.1
* **scipy**>=0.16.1
* **numpy**>=1.11.2
* **future**>=0.15.2

<br>

The **Python Requirements** can be installed using **pip install** and detailed instructions on how to install the **System Requirements** can be found in the [README file of the Github repository](https://github.com/mlampros/textTinyPy). 

<br>

The following **classes** are part of the package:



#### **classes** 


<br>


|     big_text_files           |     docs_matrix             |  token_stats                  |    tokenizer           |      utils                    |
| :--------------------------: |  :------------------------: | :---------------------------: | :--------------------: | :---------------------------: |
|   big_text_splitter()        |    Term_Matrix()            | path_2vector()                | transform_text()       | vocabulary_parser()           |
|   big_text_parser()          |    document_term_matrix()   | freq_distribution()           | transform_vec_docs()   | utf_locale()                  |
|   big_text_tokenizer()       |    term_document_matrix()   | count_character()             |                        | bytes_converter()             |
|   vocabulary_accumulator()   |    corpus_terms()           | print_count_character()       |                        | text_file_parser()            |
|                              |    Sparsity()               | collocation_words()           |                        | dice_distance()               |
|                              |    Term_Matrix_Adjust()     | print_collocations()          |                        | levenshtein_distance()        |
|                              |    most_frequent_terms()    | string_dissimilarity_matrix() |                        | cosine_distance()             |
|                              |    term_associations()      | look_up_table()               |                        | read_characters()             |
|                              |                             | print_words_lookup_tbl()      |                        | read_rows()                   |
|                              |                             |                               |                        | xml_parser_subroot_elements() |
|                              |                             |                               |                        | xml_parser_root_elements()    |
|                              |                             |                               |                        |                               |




 <br><br>
  
  
  
### *big_text_files* class
  
<br>
  
The *big_text_files* class can be utilized to process big data files and I'll illustrate this using the [english wikipedia pages and articles](https://dumps.wikimedia.org/enwiki/latest/) (to download the data use the following web-address : https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2). The size of the file (after downloading and extracting locally) is aproximalely 59.4 GB and it's of type .xml (to reproduce the results one needs to have free hard drive space of approx. 200 GB). <br> 
*Xml* files have a tree structure and one should use queries to acquire specific information. First, I'll observe the structure of the .xml file by using the utility function *read_rows()*. The *read_rows()* function takes a file as input and by specifying the *rows* argument it returns a subset of the file. It doesn't load the entire file in memory, but it just opens the file and reads the specific number of rows,

<br>
  
```py

from textTinyPy import big_text_files, docs_matrix, token_stats, tokenizer, utils


PATH = 'enwiki-latest-pages-articles.xml'         # path to file


utl = utils()                                     # initialization


subset = utl.read_rows(input_file = PATH, read_delimiter = "\n",

                       rows = 100,
                   
                       write_2file = "subs_output.txt")

```
 
 
 <br>
  
  
```py

# data subset : subs_output.txt


<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.10/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.mediawiki.org/xml/export-0.10/ http://www.mediawiki.org/xml/export-0.10.xsd" version="0.10" xml:lang="en">
  <siteinfo>
  <sitename>Wikipedia</sitename>
  <dbname>enwiki</dbname>
  <base>https://en.wikipedia.org/wiki/Main_Page</base>
  <generator>MediaWiki 1.28.0-wmf.23</generator>
  <case>first-letter</case>
  <namespaces>
  <namespace key="-2" case="first-letter">Media</namespace>
  <namespace key="-1" case="first-letter">Special</namespace>
  <namespace key="0" case="first-letter" />
  <namespace key="1" case="first-letter">Talk</namespace>
  <namespace key="2" case="first-letter">User</namespace>
  <namespace key="3" case="first-letter">User talk</namespace>
  <namespace key="4" case="first-letter">Wikipedia</namespace>
  <namespace key="5" case="first-letter">Wikipedia talk</namespace>
  <namespace key="6" case="first-letter">File</namespace>
  <namespace key="7" case="first-letter">File talk</namespace>
  <namespace key="8" case="first-letter">MediaWiki</namespace>
  .
.
.
</namespaces>
  </siteinfo>
  <page>
  <title>AccessibleComputing</title>
  <ns>0</ns>
  <id>10</id>
  <redirect title="Computer accessibility" />
  <revision>
  <id>631144794</id>
  <parentid>381202555</parentid>
  <timestamp>2014-10-26T04:50:23Z</timestamp>
  <contributor>
  <username>Paine Ellsworth</username>
  <id>9092818</id>
  </contributor>
  <comment>add [[WP:RCAT|rcat]]s</comment>
  <model>wikitext</model>
  <format>text/x-wiki</format>
  <text xml:space="preserve">#REDIRECT [[Computer accessibility]]
  
  {{Redr|move|from CamelCase|up}}</text>
  <sha1>4ro7vvppa5kmm0o1egfjztzcwd0vabw</sha1>
  </revision>
  </page>
  <page>
  <title>Anarchism</title>
  <ns>0</ns>
  <id>12</id>
  <revision>
  <id>746687538</id>
  <parentid>744318951</parentid>
  <timestamp>2016-10-28T22:43:19Z</timestamp>
  <contributor>
  <username>Eduen</username>
  <id>7527773</id>
  </contributor>
  <minor />
  <comment>/* Free love */</comment>
  <model>wikitext</model>
  <format>text/x-wiki</format>
  <text xml:space="preserve">{{Redirect2|Anarchist|Anarchists|the fictional character|Anarchist (comics)|other uses|Anarchists (disambiguation)}}
{{pp-move-indef}}

```

<br>
  
In that way one has a picture of the .xml tree structure and can continue by performing queries. The initial data file is too big to fit in the memory of a PC, thus it has to be split in smaller files, pre-processed and then returned as a single file. The main aim of the *big_text_splitter()* method is to split the data in smaller files of (approx.) equal size by either using the *batches* parameter or if the file has a structure by adding the *end_query* parameter too. Here I'll take advantage of both the *batches* and the *end_query* parameters for this task, because I'll use queries to extract the text tree-elements, so I don't want that the file is split arbitrarily. Each sub-element in the file begins and ends with the same key-word, i.e. text,

<br>
 
 
```py

bts = big_text_files()

btt = bts.big_text_splitter(input_path_file = PATH,                       # path to the enwiki data file

                            output_path_folder = "/enwiki_spl_data/",     # folder to save the files 
                            
                            batches = 40,                                 # split file in 40 batches (files) 
                            
                            end_query = '</text>',        # splits the file taking into account the key-word
                            
                            trimmed_line = False,         # the lines will be trimmed
                            
                            verbose = True)
```

<br>

**IMPORTANT NOTE**: Currently, the verbose argument works only for a Python and not for an IPython console ( [IPython only captures and displays the output at Python-level](http://stackoverflow.com/questions/29262667/how-to-allow-c-printf-to-print-in-ipython-notebook-in-cython-cell) ),

<br>


```py

approx. 10 % of data pre-processed
approx. 20 % of data pre-processed
approx. 30 % of data pre-processed
approx. 40 % of data pre-processed
approx. 50 % of data pre-processed
approx. 60 % of data pre-processed
approx. 70 % of data pre-processed
approx. 80 % of data pre-processed
approx. 90 % of data pre-processed
approx. 100 % of data pre-processed

It took 39.5592 minutes to complete the splitting

```
 
<br>

After the data is split and saved in the *output_path_folder* ("/ewiki_spl_data/") the next step is to extract the **text** tree-elements from the batches by using the *big_text_parser()* method. The latter takes as arguments the previously created *input_path_folder*, an *output_path_folder* to save the resulted text files, a *start_query*, an *end_query*, the *min_lines* (only subsets of text with more than or equal to this minimum will be kept) and the *trimmed_line* ( specifying if each line is already trimmed both-sides ),

<br>

```py


btp = bts.big_text_parser(input_path_folder = "/enwiki_spl_data/",           # the previously created folder

                          output_path_folder = "/enwiki_parse/",             # folder to save the parsed files 
                          
                          start_query = "<text xml:space=\"preserve\">",     # starts to extract text

                          end_query = "</text>",                             # stop to extract once here
                          
                          min_lines = 1, 
                          
                          trimmed_line = True,
                          
                          verbose = False)
```

 
```py

====================
batch 1 begins ...
====================

approx. 10 % of data pre-processed
approx. 20 % of data pre-processed
approx. 30 % of data pre-processed
approx. 40 % of data pre-processed
approx. 50 % of data pre-processed
approx. 60 % of data pre-processed
approx. 70 % of data pre-processed
approx. 80 % of data pre-processed
approx. 90 % of data pre-processed
approx. 100 % of data pre-processed

It took 0.291585 minutes to complete the preprocessing

It took 0.0494682 minutes to save the pre-processed data

.
.
.
.

====================
batch 40 begins ...
====================

approx. 10 % of data pre-processed
approx. 20 % of data pre-processed
approx. 30 % of data pre-processed
approx. 40 % of data pre-processed
approx. 50 % of data pre-processed
approx. 60 % of data pre-processed
approx. 70 % of data pre-processed
approx. 80 % of data pre-processed
approx. 90 % of data pre-processed
approx. 100 % of data pre-processed

It took 1.04467 minutes to complete the preprocessing

It took 0.0415537 minutes to save the pre-processed data

It took 39.0583 minutes to complete the parsing

```

<br>


Here, it's worth mentioning that the *big_text_parser* is more efficient if it extracts big chunks of text, rather than one-liners. In case of one-line text queries it has to check line by line the whole file, which is inefficient especially for files equal to the enwiki size. 

<br>
  
By extracting the text chunks from the data the .xml file size reduces to (approx.) 48.9 GB. One can now continue utilizing the *big_text_tokenizer()* method in order to tokenize and transform the data. This method takes the following parameters:
  
**batches** (each file can be further split in batches during tokenization), **to_lower** (convert to lower case), **to_upper** (convert to upper case), **LOCALE_UTF** (change utf locale depending on the language), **read_file_delimiter** (the delimiter to use for the input data, for instance a tab-delimiter or a new-line delimiter), **REMOVE_characters** (remove specific characters from the text), **remove_punctuation_string** (remove punctuation before the data is split), **remove_punctuation_vector** (remove punctuation after the data is split), **remove_numbers** (remove numbers from the data), **trim_token** (trim the tokens both-sides), **split_string** (split the string), **separator** (token split seprator where multiple delimiters can be used), **remove_stopwords** (remove stopwords using one of the available languages or by providing a user defined vector of words), **language** (the language of use), **min_num_char** (the minimum number of characters to keep), **max_num_char** (the maximum number of characters to keep), **stemmer** (stemming of the words using either the porter_2steemer or n-gram stemming -- those two methods will be explained in the tokenization function), **min_n_gram** (minimum n-grams), **max_n_gram** (maximum n-grams), **skip_n_gram** (skip n-gram), **skip_distance** (skip distance for n-grams), **n_gram_delimiter** (n-gram delimiter), **concat_delimiter** (concatenation of the data in case that one wants to save the file), **output_path_folder** (specified folder to save the data), **stemmer_ngram** (in case of n-gram stemming the n-grams), **stemmer_gamma** (in case of n-gram stemming the gamma parameter), **stemmer_truncate** (in case of n-gram stemming the truncation parameter), **stemmer_batches** (in case of n-gram stemming the batches parameter ), **threads** (the number of cores to use in parallel ), **save_2single_file** (should the output data be saved in a single file), **increment_batch_no** (the enumeration of the output files will start from this number), **vocabulary_path** (should a vocabulary be saved in a separate file), **verbose** (to print information in the console). <br>


<br>
  
More information about those parameters can be found in the package [documentation](https://mlampros.github.io/textTinyPy/_autosummary/big_text_files.html#big_text_files.big_text_files.big_text_tokenizer).

<br>
  
In this blog post I'll continue using the following transformations:

* conversion to lowercase
* trim each line
* split each line using multiple delimiters
* remove the punctuation ( once splitting is taken place )
* remove the numbers from the tokens
* limit the output words to a specific number of characters
* remove the english stopwords
* and save both the data (to a single file) and the vocabulary files (to a folder). 

Each initial file will be split in additional batches to limit the memory usage during the tokenization and transformation phase, 

<br>


```py

btok = bts.big_text_tokenizer(input_path_folder = "/enwiki_parse/",              # the previously parsed data

                              batches = 4,               # each single file will be split further in 4 batches
                              
                              to_lower = True, trim_token = True,

                              split_string=True, max_num_char = 100,
                            
                              separator = " \r\n\t.,;:()?!//",
                            
                              remove_punctuation_vector = True,
                            
                              remove_numbers = True,
                            
                              remove_stopwords = True,                
                            
                              threads = 4, 
                            
                              save_2single_file = True,      # save to a single file
                            
                              vocabulary_path = "/enwiki_vocab/",             # path to vocabulary folder
                            
                              output_path_folder="/enwiki_token/")        # folder to save the transformed data
                              
                              verbose = True)
```


```py

====================================
transformation of file 1 starts ...
====================================

-------------------
batch 1 begins ...
-------------------

input of the data starts ...
conversion to lower case starts ...
removal of numeric values starts ...
the string-trim starts ...
the split of the character string and simultaneously the removal of the punctuation in the vector starts ...
stop words of the english language will be used
the removal of stop-words starts ...
character strings with more than or equal to 1 and less than 100 characters will be kept ...
the vocabulary counts will be saved in: /enwiki_vocab/batch1.txt
the pre-processed data will be saved in a single file in: /enwiki_token/

-------------------
batch 2 begins ...
-------------------

input of the data starts ...
conversion to lower case starts ...
removal of numeric values starts ...
the string-trim starts ...
the split of the character string and simultaneously the removal of the punctuation in the vector starts ...
stop words of the english language will be used
the removal of stop-words starts ...
character strings with more than or equal to 1 and less than 100 characters will be kept ...
the vocabulary counts will be saved in: /enwiki_vocab/batch1.txt
the pre-processed data will be saved in a single file in: /enwiki_token/
.
.
.
.

====================================
transformation of file 40 starts ...
====================================

.
.
.
-------------------
batch 3 begins ...
-------------------

input of the data starts ...
conversion to lower case starts ...
removal of numeric values starts ...
the string-trim starts ...
the split of the character string and simultaneously the removal of the punctuation in the vector starts ...
stop words of the english language will be used
the removal of stop-words starts ...
character strings with more than or equal to 1 and less than 100 characters will be kept ...
the vocabulary counts will be saved in: /enwiki_vocab/batch40.txt
the pre-processed data will be saved in a single file in: /enwiki_token/

-------------------
batch 4 begins ...
-------------------

input of the data starts ...
conversion to lower case starts ...
removal of numeric values starts ...
the string-trim starts ...
the split of the character string and simultaneously the removal of the punctuation in the vector starts ...
stop words of the english language will be used
the removal of stop-words starts ...
character strings with more than or equal to 1 and less than 100 characters will be kept ...
the vocabulary counts will be saved in: /enwiki_vocab/batch40.txt
the pre-processed data will be saved in a single file in: /enwiki_token/

It took 112.115 minutes to complete tokenization

```

<br>

In total, it took approx. 191 minutes (or 3.18 hours) to pre-process (including tokenization, transformation and vocabulary extraction) the 59.4 GB of the enwiki data. <br>
 
<br>


#### **vocabulary_accumulator**

<br>

The accumulated counts of the batch vocabulary files can be computed using the **vocabulary_accumulator()** method (a word of caution : the memory consumption when running the *vocabulary_accumulator* method for this kind of data size can exceed the 10 GB),

<br>

```py


voc_acc = bts.vocabulary_accumulator(input_path_folder = "/enwiki_vocab/", 
                            
                                     output_path_file = "/VOCAB.txt",
                            
                                     max_num_chars = 50,
                                     
                                     verbose = True)
```


```py

vocabulary.of.batch 1 will.be.merged ...
vocabulary.of.batch 2 will.be.merged ...
vocabulary.of.batch 3 will.be.merged ...
.
.
.
vocabulary.of.batch 39 will.be.merged ...
vocabulary.of.batch 40 will.be.merged ...	minutes.to.merge.sort.batches: 4.98656
	minutes.to.save.data: 0.49200

```

<br>

The following table shows the first rows of the vocabulary counts,

<br>


|    terms   |   frequency  |  
| :--------: |  :---------: | 
|  lt        |   111408435L | 
|  refgt     |   49197149L  | 
|  quot      |   48688082L  |
|  gt        |   47466149L  |
|  user      |   32042007L  |
|  category  |   30619748L  |
|  www       |   25358252L  |
|  http      |   23008243L  |



<br>

#### **word vectors**

<br>

Having a clean single text file of all the wikipedia pages and articles one can perform many tasks, for instance one can compute **word vectors**.
"A [word representation](http://www.iro.umontreal.ca/~lisa/pointeurs/turian-wordrepresentations-acl10.pdf) is a mathematical object associated with each word, often a vector. Each dimension's value corresponds to a feature and might even have a semantic or grammatical interpretation, so we call it a word feature. Conventionally, supervised lexicalized NLP approaches take a word and convert it to a symbolic ID, which is then transformed into a feature vector using a one-hot representation: The feature vector has the same length as the size of the vocabulary, and only one dimension is on." <br>
One of the options to compute word vectors in python is by using the [fasttext](https://github.com/salestock/fastText.py) package, which is a Python interface for the Facebook fastText library.

Currently, there are many resources on the web on how to use [pre-trained word vectors (embeddings) as input to neural networks](https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html).

<br>

For illustration purposes I'll limit the train data to approx. 1 GB of the output file,

<br>

```py


reduced_data = utl.read_characters(input_file = "/output_token_single_file.txt", 

                                   characters = 1000000000,                     # approx. 1 GB of the data
                                   
                                   write_2file = "/reduced_single_file.txt")


import fasttext


model = fasttext.skipgram(input_file = "/reduced_single_file.txt",       # Skipgram model for the reduced data
          
                          output = 'model',
                          
                          dim = 50,                                      # 50-dimensional word vectors
                          
                          thread = 4,
                          
                          silent = 0)

```

```py
.
.
Progress: 99.6%  words/sec/thread: 33739  lr: 0.000198  loss: 0.193076  eta: 0h0m
.
.
Progress: 100.0%  words/sec/thread: 33734  lr: 0.000000  loss: 0.192349  eta: 0h0m 
Progress: 100.0%  words/sec/thread: 33734  lr: 0.000000  loss: 0.192349  eta: 0h0m 

```

<br>

the following vector-subset is the example output of the "model.vec" file, which includes the 50-dimensional word vectors, the first line gives the resulted dimensions of the word vectors (here 620255 50),

<br>

```py

620255 50
lt 0.0018994 -0.19756 -0.10526 -0.32816 -0.35156 -0.02187 -0.13948 0.15977 0.5009 0.27858 .........
refgt 0.20077 -0.26631 0.0035167 -0.17726 -0.14656 -0.09296 -0.46368 0.1134 0.35207 0.30549  .........
quot 0.38134 0.12661 -0.0016454 -0.3684 -0.19769 -0.26847 -0.24503 -0.39303 0.92669 0.37607  .........
gt 0.11586 -0.099941 -0.27291 -0.28185 -0.6023 0.021851 -0.34048 -0.09962 0.6575 0.41988  .........
cite 0.36266 -0.13352 0.031717 -0.27373 -0.42416 0.030359 -0.53785 0.26502 0.55605 0.33584  .........
www 0.11635 -0.44419 0.22903 -0.16716 -0.49226 -0.34623 -0.30852 -0.16122 -0.032654 0.38568  .........
ref 0.38567 0.12687 -0.16743 -0.32466 -0.47925 0.1354 -0.36213 -0.021718 0.52973 0.50027  .........
http 0.087524 -0.48365 0.28409 -0.14553 -0.25343 -0.40087 -0.29781 -0.3642 -0.29402 0.34416  .........
namequot 0.24773 0.14029 -0.21957 -0.30853 -0.61345 -0.072705 -0.39911 -0.12125 0.56848  ......... 
– -0.23966 -0.14057 0.18155 -0.33832 -0.17367 0.29058 -0.50431 -0.12717 0.4548 0.11858  .........
amp -0.60338 0.024924 0.22195 -0.66071 0.079215 -0.15774 -0.13916 -0.26569 0.68348  .........
category 0.85996 -0.32525 0.63248 -0.99685 -0.96834 0.13842 -1.1631 -1.2552 0.032192 -0.47464  .........
county 0.10389 -0.39652 0.79461 0.10313 0.46214 0.77049 0.36431 0.70811 0.12155 -0.7107  .........
org 0.12906 -1.0284 0.060272 -0.20488 -0.2518 -0.25462 -0.32675 -0.17685 -0.22979 0.11107 .........
states 0.20618 0.13609 0.58467 0.014138 0.32926 -0.049853 -0.24458 0.21669 0.5693 0.18564  .........
united 0.41388 0.19653 0.35726 -0.011942 0.30908 -0.027071 -0.41922 0.18339 0.54227  .........
web 0.44993 -0.71661 0.079289 -0.048054 -0.53383 -0.19032 -0.47208 -0.15838 0.0077261  .........
census -0.15643 -0.36542 0.774 -0.41559 0.89984 0.45845 0.89599 -0.38108 0.61417 -0.36313  .........
.
.
.
.

```



<br><br>
  
### *docs_matrix* class

<br>

The *docs_matrix* class includes methods for building a document-term or a term-document matrix and extracting information from those matrices (it is based on the Armadillo library and wrapped using the scipy library). I'll explain all the different methods using a toy text file downloaded from wikipedia,

<br>

```R

The term planet is ancient, with ties to history, astrology, science, mythology, and religion. Several planets in the Solar System can be seen with the naked eye. These were regarded by many early cultures as divine, or as emissaries of deities. As scientific knowledge advanced, human perception of the planets changed, incorporating a number of disparate objects. In 2006, the International Astronomical Union (IAU) officially adopted a resolution defining planets within the Solar System. This definition is controversial because it excludes many objects of planetary mass based on where or what they orbit. 
Although eight of the planetary bodies discovered before 1950 remain planets under the modern definition, some celestial bodies, such as Ceres, Pallas, Juno and Vesta (each an object in the solar asteroid belt), and Pluto (the first trans-Neptunian object discovered), that were once considered planets by the scientific community, are no longer viewed as such.
The planets were thought by Ptolemy to orbit Earth in deferent and epicycle motions. Although the idea that the planets orbited the Sun had been suggested many times, it was not until the 17th century that this view was supported by evidence from the first telescopic astronomical observations, performed by Galileo Galilei. 
At about the same time, by careful analysis of pre-telescopic observation data collected by Tycho Brahe, Johannes Kepler found the planets orbits were not circular but elliptical. As observational tools improved, astronomers saw that, like Earth, the planets rotated around tilted axes, and some shared such features as ice caps and seasons. Since the dawn of the Space Age, close observation by space probes has found that Earth and the other planets share characteristics such as volcanism, hurricanes, tectonics, and even hydrology.
Planets are generally divided into two main types: large low-density giant planets, and smaller rocky terrestrials. Under IAU definitions, there are eight planets in the Solar System. In order of increasing distance from the Sun, they are the four terrestrials, Mercury, Venus, Earth, and Mars, then the four giant planets, Jupiter, Saturn, Uranus, and Neptune. Six of the planets are orbited by one or more natural satellites.

```

<br>

The *docs_matrix* class can be initialized using either a vector of documents or a text file. Assuming the downloaded file is saved as "planets.txt", then a document-term-matrix can be created in the following way,

<br>



```py

dcm = docs_matrix()

dtm = dcm.Term_Matrix(vector_documents = None, 

                     path_2documents_file = "/planets.txt", 
                     
                     sort_terms = True, 
                     
                     to_lower = True,                       # convert to lower case

                     trim_token = True,                     # trim token

                     split_string = True,                   # split string

                     tf_idf = True,                         # tf-idf will be returned

                     verbose = True)
```


```py

dtm_out = dcm.document_term_matrix(to_array = True)                     # returns dtm as array

warning: the following terms sum to zero :  ['of', 'were', 'in']

```


```py

dtm_out                        # the output array


array([[-0.00193959,  0.        ,  0.        , ...,  0.00974777,
         0.01949555,  0.00974777],
       [-0.00325574,  0.        ,  0.01636233, ...,  0.        ,
         0.        ,  0.        ],
       [-0.00344003,  0.0172885 ,  0.        , ...,  0.        ,
         0.        ,  0.        ],
       [-0.00219665,  0.        ,  0.        , ...,  0.        ,
         0.        ,  0.        ],
       [-0.0026812 ,  0.        ,  0.        , ...,  0.        ,
         0.        ,  0.        ]])
         
```

```py

dtm_out.shape                 # output shape of the array

(5, 212)

```

```py

dcm.corpus_terms()           # corpus terms (columns of the output document matrix)


array(['', '17th', '1950', '2006', 'a', 'about', 'adopted', 'advanced',
       'age', 'although', 'an', 'analysis', 'ancient', 'and', 'are',
       'around', 'as', 'asteroid', 'astrology', 'astronomers',
       'astronomical', 'at', 'axes', 'based', 'be', 'because', 'been',
       'before', 'belt', 'bodies', 'brahe', 'but', 'by', 'can', 'caps',
       'careful', 'celestial', 'century', 'ceres', 'changed',
        .
        .
        .
        .
       'the', 'then', 'there', 'these', 'they', 'this', 'thought', 'ties',
       'tilted', 'time', 'times', 'to', 'tools', 'trans-neptunian', 'two',
       'tycho', 'types', 'under', 'union', 'until', 'uranus', 'venus',
       'vesta', 'view', 'viewed', 'volcanism', 'was', 'were', 'what',
       'where', 'with', 'within'], 
      dtype='|S15')

```

```py

dcm.Sparsity()                  # outputs the sparsity of the dtm

'sparsity of the matrix: 75.3774 %'

```


<br>

The *Term_Matrix* method takes almost the same parameters as the ( already explained ) *big_text_tokenizer()* method. The only differences are: 

* **sort_terms** ( should the output terms - rows or columns depending on the *document_term_matrix* parameter - be sorted in alphabetical order )
* **print_every_rows** ( verbose output intervalls )
* **normalize** ( applies *l1* or *l2* normalization )
* **tf_idf** ( the term-frequency-inverse-document-frequency will be returned )


Details about the parameters of the *Term_Matrix* method can be found in the package [documentation](https://mlampros.github.io/textTinyPy/_autosummary/docs_matrix.html#docs_matrix.docs_matrix). <br>

<br> 

To adjust the sparsity of the output matrix one can take advantage of the *Term_Matrix_Adjust* method, (by adjusting the *sparsity_thresh* parameter towards 0.0 a proportion of the sparse terms will be removed),

<br>

```py

dtm_adj = dcm.Term_Matrix_Adjust(sparsity_thresh = 0.6)    

warning: the following terms sum to zero :  ['of', 'were', 'in']         

```

<br>

and a warning will be printed for the terms (here columns) that sum to zero.

<br>


```py

dtm_adj                  # output of the adjusted array


array([[-0.00581877, -0.00193959,  0.        , -0.00193959,  0.00474774,
         0.        , -0.00193959,  0.0071216 , -0.01163755,  0.        ,
         0.        ,  0.        ],
       [-0.00651148, -0.00325574,  0.        , -0.00325574,  0.00398471,
         0.        , -0.00651148,  0.00796941, -0.01627871,  0.        ,
         0.00398471,  0.        ],
       [-0.00688006, -0.01032009,  0.        , -0.00344003,  0.        ,
         0.        , -0.00344003,  0.        , -0.02064018,  0.        ,
         0.00842051,  0.00421026],
       [-0.00658994, -0.00658994,  0.        , -0.00219665,  0.        ,
         0.        , -0.00878658,  0.00806543, -0.01317987,  0.        ,
         0.00537695,  0.00537695],
       [-0.013406  , -0.0026812 ,  0.        , -0.0026812 ,  0.00328152,
         0.        , -0.0080436 ,  0.        , -0.013406  ,  0.        ,
         0.        ,  0.00328152]])

````


<br>

The *term_associations* method returns the correlation of specific terms (*Terms*) with all the other terms in the output matrix. The dimensions of the output matrix depends on which one of the *Term_Matrix*, *Term_Matrix_Adjust* I run previously. In the previous step I adjusted the initial sparse matrix using a sparsity_thresh of 0.6, thus the new dimensions will be,

<br>

```py

 dtm_adj.shape
 
(5, 12)

```
<br>

and the resulted terms,

```py

dcm.corpus_terms()

array(['planets', 'by', 'of', '', 'solar', 'were', 'and', 'as', 'the',
       'in', 'that', 'earth'], 
      dtype='|S7')

```

<br>

The associations for the terms ['planets', 'by', 'INVALID'] are,

<br>

```py

asc = dcm.term_associations(Terms = ['planets', 'by', 'INVALID'], keep_terms = None, verbose = True)

```

<br>

in case that one of the terms is not present in the corpus then a warning will be printed in the console, and for zero-valued columns NaN will be returned,  

<br>


```py

the 'INVALID' term does not exist in the terms list

total.number.variables.processed:   1
total.number.variables.processed:   2	minutes.to.complete: 0.00000


asc                                  # output of the function


{'by':        term  correlation
1   planets    -0.263013
2        of          NaN
3     solar     0.909278
4               0.501003
5      were          NaN
6       the     0.783844
7        as     0.369824
8       and    -0.059415
9        in          NaN
10    earth    -0.695276
11     that    -0.933888

[11 rows x 2 columns], 'planets':      term  correlation
1      by    -0.263013
2      of          NaN
3             0.075218
4   solar    -0.118875
5    were          NaN
6      as     0.659432
7     and     0.482527
8     the    -0.155009
9      in          NaN
10   that     0.443076
11  earth    -0.242261

[11 rows x 2 columns]}

```

<br>

Lastly, the *most_frequent_terms* method gives the frequency of the terms in the corpus. However, the function returns only if the *normalize* parameter is None and the *tf_idf* parameter is False ( the latter two parameters belong to the *Term_Matrix()* method ),

<br>


```py

dcm = docs_matrix()                                 #  run once again with tf_idf = False

dtm = dcm.Term_Matrix(vector_documents = None, 

                     path_2documents_file = "/planets.txt", 
                     
                     sort_terms = True, 
                     
                     to_lower = True,                       # convert to lower case

                     trim_token = True,                     # trim token

                     split_string = True,                   # split string

                     tf_idf = False,                        # disable tf-idf

                     verbose = True)
                     
                     
mft = dcm.most_frequent_terms(keep_terms = 10,              # keep only first 10 terms

                              threads = 1, 
                        
                              verbose = True)     
```

```py

minutes.to.complete: 0.00000     

      terms  frequency
0      the         28
1  planets         15
2      and         11
3       of          9
4       by          9
5       as          8
6       in          6
7                   5
8      are          5
9     that          5                    
                     
                     
```

<br>

More information about the *docs_matrix* class can be found in the package [documentation](https://mlampros.github.io/textTinyPy/_autosummary/docs_matrix.html#).

<br><br>


### *token_stats* class

<br>

The *token_stats* class can be utilized to output corpus statistics. Each of the following methods can take either a *list of terms*, a *text file* or *a folder of files* as input:

<br>

* **path_2vector** : is a helper method which takes a path to a file or folder of files and returns the content in form of a list,

<br>

```py

tks = token_stats()

tks.path_2vector(path_2folder = None, 

                 path_2file = "/planets.txt",
                 
                 file_delimiter = "\n")
 
```

```py

array([ 'The term planet is ancient, with ties to history, astrology, science, mythology, and religion. Several planets in the Solar System can be seen with the naked eye. These were regarded by many early cultures as divine, or as emissaries of deities. As scientific knowledge advanced, human perception of the planets changed, incorporating a number of disparate objects. In 2006, the International Astronomical Union (IAU) officially adopted a resolution defining planets within the Solar System. This definition is controversial because it excludes many objects of planetary mass based on where or what they orbit. ',
       'Although eight of the planetary bodies discovered before 1950 remain planets under the modern definition, some celestial bodies, such as Ceres, Pallas, Juno and Vesta (each an object in the solar asteroid belt), and Pluto (the first trans-Neptunian object discovered), that were once considered planets by the scientific community, are no longer viewed as such.',
       .
       .
       .

'Planets are generally divided into two main types: large lowdensity giant planets, and smaller rocky terrestrials. Under IAU definitions, there are eight planets in the Solar System. In order of increasing distance from the Sun, they are the four terrestrials, Mercury, Venus, Earth, and Mars, then the four giant planets, Jupiter, Saturn, Uranus, and Neptune. Six of the planets are orbited by one or more natural satellites.'], 
      dtype='|S611')
```


<br>

* **freq_distribution** : it returns a named-unsorted list frequency distribution for a vocabulary file

<br>

```py

# assuming the following 'freq_vocab.txt'

the
term
planet
is
ancient
with
ties
to
history
astrology
science
mythology
and
religion
several
planets
in
the
solar
system
can
be
seen
with
the
naked
eye
these
were

```

<br>

this method would return,

<br>

```py

tks.freq_distribution(x_vector = None, 

                      path_2folder = None, 
                      
                      path_2file = 'freq_vocab.txt', 
                      
                      file_delimiter = "\n", keep = 10)
```


```py
           freq
the           3
with          2
history       1
religion      1
is            1
planets       1
in            1
seen          1
astrology     1
naked         1

[10 rows x 1 columns]

```

<br>

* **count_character** : it returns the number of characters for each word of the corpus. 

<br>

for the previously mentioned *'freq_vocab.txt'* it would output,

<br>

```py

res_cnt = tks.count_character(x_vector = None, 

                              path_2folder = None, 
                              
                              path_2file = 'freq_vocab.txt', 
                              
                              file_delimiter = "\n")
```

```py

cnt


array([2, 3, 4, 5, 6, 7, 8, 9])

```

```py

# words with number of characters equal to 3

tks.print_count_character(number = 3)

```


```py

array(['the', 'and', 'the', 'can', 'the', 'eye'], 
      dtype='|S3')

```

<br>

* **collocation_words** : it returns a co-occurence frequency table for n-grams. "A [collocation]( http://nlp.stanford.edu/fsnlp/promo/colloc.pdf) is defined as a sequence of two or more consecutive words, that has characteristics of a syntactic and semantic unit, and whose exact and unambiguous meaning or connotation cannot be derived directly from the meaning or connotation of its components". The input to the function should be text n-grams separated by a delimiter ( for instance the *transform_text()* function in the next code chunk will build n_grams of length 3 ),

<br>

```py

# "planets.txt" file as input

tok = tokenizer()

trans = tok.transform_text(input_string = "planets.txt",

                           to_lower = True, 
                              
                           split_string = True,
                          
                           min_n_gram = 3, 
                          
                           max_n_gram = 3, 
                          
                           n_gram_delimiter = "_")
```


```py

trans                           # example output

['the_term_planet', 'term_planet_is', 'planet_is_ancient', 'is_ancient_with', 'ancient_with_ties', 'with_ties_to', 'ties_to_history', 'to_history_astrology', 'history_astrology_science', 'astrology_science_mythology', 'science_mythology_and', 'mythology_and_religion', 'and_religion_several', 'religion_several_planets', 'several_planets_in', 'planets_in_the', 'in_the_solar', 'the_solar_system', 
.
.
.
'four_giant_planets', 'giant_planets_jupiter', 'planets_jupiter_saturn', 'jupiter_saturn_uranus', 'saturn_uranus_and', 'uranus_and_neptune', 'and_neptune_six', 'neptune_six_of', 'six_of_the', 'of_the_planets', 'the_planets_are', 'planets_are_orbited', 'are_orbited_by', 'orbited_by_one', 'by_one_or', 'one_or_more', 'or_more_natural', 'more_natural_satellites', 'natural_satellites_']

```

```py

col_lst = tks.collocation_words(x_vector = trans,           # takes the n-grams as input

                                file_delimiter = "\n",
                               
                                n_gram_delimiter = "_")

```

```py

col_lst                    # example output


array(['', '17th', '1950', '2006', 'a', 'about', 'adopted', 'advanced',
       'age', 'although', 'an', 'analysis', 'ancient', 'and', 'are',
       'around', 'as', 'asteroid', 'astrology', 'astronomers',
       'astronomical', 'at', 'axes', 'based', 'be', 'because', 'been',
       'before', 'belt', 'bodies', 'brahe', 'but', 'by', 'can', 'caps',
        .
        .
        .
        .
       'tycho', 'types', 'under', 'union', 'until', 'uranus', 'venus',
       'vesta', 'view', 'viewed', 'volcanism', 'was', 'were', 'what',
       'where', 'with', 'within'], 
      dtype='|S15')

```

<br>

and the *print_collocations* method returns the collocations for the example word *ancient*,

<br>

```py

res = tks.print_collocations(word = "ancient")

```

```py

res


{'planet': 0.167, 'ties': 0.167, 'is': 0.333, 'with': 0.333}

```

<br>

* **string_dissimilarity_matrix** : it returns a string-dissimilarity-matrix using either the *dice*, *levenshtein* or *cosine distance*. The input can be a *character string list* only. In case that the method is dice then the dice-coefficient (similarity) is calculated between two strings for a specific number of character n-grams ( dice_n_gram ). The *dice* and *levenshtein* methods are applied to words, whereas the *cosine* distance to word-sentences.

<br>

For illustration purposes I'll use the previously mentioned *'freq_vocab.txt'* file, but first I have to convert the text file to a character vector,

<br>
  
```py

tks_init = token_stats()

tmp_vec = tks_init.path_2vector(path_2file = 'freq_vocab.txt')

```


```py

tmp_vec           # output

array(['the', 'term', 'planet', 'is', 'ancient', 'with', 'ties', 'to',
       'history', 'astrology', 'science', 'mythology', 'and', 'religion',
       'several', 'planets', 'in', 'the', 'solar', 'system', 'can', 'be',
       'seen', 'with', 'the', 'naked', 'eye', 'these', 'were'], 
      dtype='|S9')
```


```py

res = tks.string_dissimilarity_matrix(words_vector = list(tmp_vec),              # takes as input the list

                                      dice_n_gram = 2, method = "dice", 
                                      
                                      split_separator = " ", dice_thresh = 1.0, 
                                           
                                      upper = True, diagonal = True, threads = 1)
```


```py

res                          # example output

            ancient       and  astrology  be  can  eye   history  in  is     ....
ancient    0.000000  0.500000   0.857143   1    1    1  1.000000   1   1     ....
and        0.500000  0.000000   0.800000   1    1    1  1.000000   1   1     ....
astrology  0.857143  0.800000   0.000000   1    1    1  0.857143   1   1     ....
be         1.000000  1.000000   1.000000   0    1    1  1.000000   0   0     ....
can        1.000000  1.000000   1.000000   1    0    1  1.000000   1   1     ....
eye        1.000000  1.000000   1.000000   1    1    0  1.000000   1   1     ....
history    1.000000  1.000000   0.857143   1    1    1  0.000000   1   1     ....
in         1.000000  1.000000   1.000000   0    1    1  1.000000   0   0     ....
is         1.000000  1.000000   1.000000   0    1    1  1.000000   0   0     ....
mythology  1.000000  1.000000   0.625000   1    1    1  1.000000   1   1     ....
naked      1.000000  1.000000   1.000000   1    1    1  1.000000   1   1     ....

.
.
.

```

<br>
  
here by adjusting (reducing ) the *dice_thresh* parameter we can force values close to 1.0 to become 1.0,

<br>

```py

res = tks.string_dissimilarity_matrix(words_vector = list(tmp_vec),              # takes as input the list

                                      dice_n_gram = 2, method = "dice", 
                                      
                                      split_separator = " ", dice_thresh = 0.5, 
                                           
                                      upper = True, diagonal = True, threads = 1)

```

```py

            ancient  and  astrology  be  can  eye  history  in  is  mythology       .... 
ancient    0.000000    1          1   1    1    1        1   1   1          1       .... 
and        1.000000    0          1   1    1    1        1   1   1          1       .... 
astrology  1.000000    1          0   1    1    1        1   1   1          1       .... 
be         1.000000    1          1   0    1    1        1   0   0          1       .... 
can        1.000000    1          1   1    0    1        1   1   1          1       .... 
eye        1.000000    1          1   1    1    0        1   1   1          1       .... 
history    1.000000    1          1   1    1    1        0   1   1          1       .... 
in         1.000000    1          1   0    1    1        1   0   0          1       .... 
is         1.000000    1          1   0    1    1        1   0   0          1       .... 
mythology  1.000000    1          1   1    1    1        1   1   1          0       .... 
naked      1.000000    1          1   1    1    1        1   1   1          1       .... 

.
.
.

```


<br>
  
* **look_up_table** : The idea here is to split the input words to n-grams using a numeric value and then retrieve the words which have a similar character n-gram. <br>
It returns a look-up-list where the list-names are the n-grams and the dictionaries are the words associated with those n-grams. The words for each n-gram can be retrieved using the *print_words_lookup_tbl* method. The input can be a character string list only.

<br>
  
```py

lkt = tks.look_up_table(words_vector = list(tmp_vec),       # use of the previously created array

                        n_grams = 3)

```


```py

lkt              # example output


array(['', '_an', '_as', '_hi', '_my', '_na', '_pl', '_re', '_sc', '_se',
       '_so', '_sy', '_te', '_th', '_ti', '_we', '_wi', 'ake', 'anc',
       'ane', 'ast', 'cie', 'eli', 'enc', 'era', 'eve', 'gio', 'hes',
       'his', 'hol', 'ien', 'igi', 'ist', 'lan', 'lig', 'log', 'myt',
       'nak', 'nci', 'net', 'ola', 'olo', 'pla', 'rel', 'rol', 'sci',
       'see', 'sev', 'sol', 'ste', 'sto', 'str', 'sys', 'ter', 'the',
       'tho', 'tie', 'tor', 'tro', 'ver', 'wer', 'wit', 'yst', 'yth'], 
      dtype='|S3')
```

<br>
  
then retrieve words with same n-grams,

<br>
  
```py

tks.print_words_lookup_tbl(n_gram = "log")

```


```py

array(['_astrology_', '_mythology_'], 
      dtype='|S11')

```

<br>
  
the underscores are necessary to distinguish the begin and end of each word when computing the n-grams.

More information about the *token_stats* class can be found in the package [documentation](https://mlampros.github.io/textTinyPy/_autosummary/token_stats.html).

<br><br>


### *tokenizer* class
  
<br>
  
The **transform_text()** method applies tokenization and transformation in a similar way to the *big_text_tokenizer()* method, however for small to medium data sets. The input can be either a character string (text data) or a path to a file. This method takes as input a single character string (character-string == of length one). The parameters for the *transform_text()* method are the same to the (already explained) *big_text_tokenizer()* method with the only exception being the input data type.

<br>
  
The **transform_vec_docs()** function works in the same way to the *Term_Matrix()* method and it targets small to medium data sets. It takes as input a vector of documents and retains their order after tokenization and transformation has taken place. Both the *transform_text()* and *transform_vec_docs()* share the same parameters, with the following two exceptions,

* the input of the *transform_text()* method is an **input_string**, whereas for the *transform_vec_docs()* method is an **input_list**
* the **as_token** parameter applies only to the *transform_vec_docs()* method : if True then the output of the function is a list of lists (of split token). Otherwise it's a list of character strings (sentences)

<br>

The following code chunks give an overview of the mentioned functions,

<br>


```py

#----------------
# transform_text
#----------------


# example input : "planets.txt"



init_trans_tok = tokenizer()


res_txt = init_trans_tok.transform_text(input_string = "/planets.txt", 
  
                                        to_lower = True,
      
                                        LOCALE_UTF = "",           
                                        
                                        trim_token = True,
                                        
                                        split_string = True,
                                        
                                        remove_stopwords = True, 
                                        
                                        language = "english",
                                        
                                        stemmer = "ngram_sequential",

                                        stemmer_ngram = 3,

                                        threads = 1)
```
<br>

the output is a vector of *tokens* after the english stopwords were removed and the terms were stemmed (*ngram_sequential* of length 3),

<br>

```py

res_txt             # example output


['ter', 'planet', 'anci', 'ties', 'hist', 'astro', 'scien', 'mythol',
 'relig', 'planet', 'solar', 'system', 'naked', 'eye', 'regar',
 'early', 'cultu', 'divi', 'emissar', 'deit', 'scien', 'knowle',
 'advan', 'human', 'percept', 'planet', 'chan', 'incorporat',
 'number', 'dispar', 'object', '2006', 'internatio', 'astro',
 'union', 'iau', 'officia', 'adop', 'resolut', 'defini', 'planet',
 'solar', 'system', 'defini', 'controvers', 'exclu', 'object',
 'planet', 'mass', 'based', 'orbit', 'planet', 'bodies', 'discove',
 '1950', 'remain', '"plane', 'modern', 'defini', 'celest', 'bodies',
 'ceres', 'pallas', 'juno', 'vesta', 'object', 'solar', 'aster',
 'belt', 'pluto', 'trans-neptun', 'object', 'discove', 'conside',
 'planet', 'scien', 'commun', 'longer', 'view', 'planet', 'thou',
 'ptol', 'orbit', 'earth', 'defer', 'epicy', 'moti', 'idea',
 'planet', 'orbit', 'sun', 'sugges', 'time', '17th', 'cent', 'view',
 'suppor', 'evide', 'telesco', 'astro', 'observation', 'perfor',
 'galile', 'galile', 'time', 'care', 'analy', 'pre-telesco',
 'observat', 'data', 'collec', 'tycho', 'brahe', 'johan', 'kepler',
 'found', 'planet', 'orbit', 'circu', 'ellipti', 'observation',
 'tools', 'impro', 'astro', 'earth', 'planet', 'rota', 'tilted',
 'axes', 'share', 'featu', 'ice', 'caps', 'seas', 'dawn', 'space',
 'age', 'close', 'observat', 'space', 'probes', 'found', 'earth',
 'planet', 'share', 'characterist', 'volcan', 'hurrica', 'tecton',
 'hydrol', 'planet', 'genera', 'divi', 'main', 'types', 'large',
 'low-dens', 'giant', 'planet', 'smal', 'rocky', 'terrestri', 'iau',
 'defini', 'planet', 'solar', 'system', 'order', 'increas', 'dista',
 'sun', 'terrestri', 'merc', 'venus', 'earth', 'mars', 'giant',
 'planet', 'jupi', 'saturn', 'uranus', 'nept', 'planet', 'orbit',
 'natu', 'satelli']

```


```py

#--------------------
# transform_vec_docs
#--------------------


# the input should be a vector of documents

tks = token_stats()

vec_docs = tks.path_2vector(path_2file = "/planets.txt", file_delimiter = "\n")

```

```py

vec_docs
array([ 'The term planet is ancient, with ties to history, astrology, science, mythology, and religion. Several planets in the Solar System can be seen with the naked eye. These were regarded by many early cultures as divine, or as emissaries of deities. As scientific knowledge advanced, human perception of the planets changed, incorporating a number of disparate objects. In 2006, the International Astronomical Union (IAU) officially adopted a resolution defining planets within the Solar System. This definition is controversial because it excludes many objects of planetary mass based on where or what they orbit. ',
.
.
.

  'Planets are generally divided into two main types: large lowdensity giant planets, and smaller rocky terrestrials. Under IAU definitions, there are eight planets in the Solar System. In order of increasing distance from the Sun, they are the four terrestrials, Mercury, Venus, Earth, and Mars, then the four giant planets, Jupiter, Saturn, Uranus, and Neptune. Six of the planets are orbited by one or more natural satellites.'], 
      dtype='|S611')

```


```py

res_dct = init_trans_tok.transform_vec_docs(input_list = list(vec_docs),       # input is list of documents 

                                            as_token = False,                  # return character vector of documents
                                            
                                            to_lower = True,
                                            
                                            LOCALE_UTF = "",           
                                            
                                            trim_token = True,
                                            
                                            split_string = True,
                                            
                                            remove_stopwords = True, 
                                            
                                            language = "english",
                                      
                                            stemmer = "porter2_stemmer", 
                                  
                                            threads = 1)
```

<br>

the output is a list (of equal length as the vec_docs array) of *transformed documents* after the english stopwords were removed and the terms were stemmed (*porter2-stemming*),

<br>

```py

res_dct             # example output

['term planet ancient tie histori astrolog scienc mytholog religion planet solar system nake eye regard ear cultur divin emissari deiti scientif knowledg advanc human percept planet chang incorpor number dispar object 2006 intern astronom union iau offici adopt resolut defin planet solar system definit controversi exclud object planetari mass base orbit',
'planetari bodi discov 1950 remain "planets" modern definit celesti bodi cere palla juno vesta object solar asteroid belt pluto trans-neptunian object discov consid planet scientif communiti longer view', 
'planet thought ptolemi orbit earth defer epicycl motion idea planet orbit sun suggest time 17th centuri view support evid telescop astronom observ perform galileo galilei', 
'time care analysi pre-telescop observ data collect tycho brahe johann kepler found planet orbit circular ellipt observ tool improv astronom earth planet rotat tilt axe share featur ice cap season dawn space age close observ space probe found earth planet share characterist volcan hurrican tecton hydrolog', 
'planet general divid main type larg low-dens giant planet smaller rocki terrestri iau definit planet solar system order increas distanc sun terrestri mercuri venus earth mar giant planet jupit saturn uranus neptun planet orbit natur satellit']

```

<br>

The documents can be returned as a list of lists by specifying, *as_token* = True,

<br>

```py

res_dct = init_trans_tok.transform_vec_docs(input_list = list(vec_docs),       # input is list of documents 

                                            as_token = True,                  # return character vector of documents
                                            
                                            to_lower = True,
                                            
                                            LOCALE_UTF = "",           
                                            
                                            trim_token = True,
                                            
                                            split_string = True,
                                            
                                            remove_stopwords = True, 
                                            
                                            language = "english",
                                            
                                            stemmer = "porter2_stemmer", 
                                  
                                            threads = 1)

```

<br>

```py

res_dct               # example output


[ ['term', 'planet', 'ancient', 'tie', 'histori', 'astrolog', 'scienc', 'mytholog', 'religion', 'planet', 'solar', 'system', 'nake', 'eye', 'regard', 'ear', 'cultur', 'divin', 'emissari', 'deiti', 'scientif', 'knowledg', 'advanc', 'human', 'percept', 'planet', 'chang', 'incorpor', 'number', 'dispar', 'object', '2006', 'intern', 'astronom', 'union', 'iau', 'offici', 'adopt', 'resolut', 'defin', 'planet', 'solar', 'system', 'definit', 'controversi', 'exclud', 'object', 'planetari', 'mass', 'base', 'orbit'],
['planetari', 'bodi', 'discov', '1950', 'remain', '"planets"', 'modern', 'definit', 'celesti', 'bodi', 'cere', 'palla', 'juno', 'vesta', 'object', 'solar', 'asteroid', 'belt', 'pluto', 'trans-neptunian', 'object', 'discov', 'consid', 'planet', 'scientif', 'communiti', 'longer', 'view'],
['planet', 'thought', 'ptolemi', 'orbit', 'earth', 'defer', 'epicycl', 'motion', 'idea', 'planet', 'orbit', 'sun', 'suggest', 'time', '17th', 'centuri', 'view', 'support', 'evid', 'telescop', 'astronom', 'observ', 'perform', 'galileo', 'galilei'],
['time', 'care', 'analysi', 'pre-telescop', 'observ', 'data', 'collect', 'tycho', 'brahe', 'johann', 'kepler', 'found', 'planet', 'orbit', 'circular', 'ellipt', 'observ', 'tool', 'improv', 'astronom', 'earth', 'planet', 'rotat', 'tilt', 'axe', 'share', 'featur', 'ice', 'cap', 'season', 'dawn', 'space', 'age', 'close', 'observ', 'space', 'probe', 'found', 'earth', 'planet', 'share', 'characterist', 'volcan', 'hurrican', 'tecton', 'hydrolog'],
['planet', 'general', 'divid', 'main', 'type', 'larg', 'low-dens', 'giant', 'planet', 'smaller', 'rocki', 'terrestri', 'iau', 'definit', 'planet', 'solar', 'system', 'order', 'increas', 'distanc', 'sun', 'terrestri', 'mercuri', 'venus', 'earth', 'mar', 'giant', 'planet', 'jupit', 'saturn', 'uranus', 'neptun', 'planet', 'orbit', 'natur', 'satellit']]

```


<br>

A few words about the **LOCALE_UTF**, **remove_stopwords** and **stemmer** parameters.

<br>

* The **LOCALE_UTF** can take as input either an empty string ("") or a character string (for instance "el_GR.UTF-8"). It should be a non-empty string if the text input is other than english.

<br>

* The **remove_stopwords** parameter can be either a boolean (True, False) or a character vector of user defined stop-words. The available languages are specified by the parameter **language**. Currently, there is no support for chinese, japanese, korean, thai or languages with ambiguous word boundaries.

<br>

* The **stemmer** parameter can take as input one of the **porter2_stemmer**, **ngram_sequential** or **ngram_overlap**. 
    + The [*porter2_stemmer*](https://github.com/smassung/porter2_stemmer) is a C++ implementation of the [snowball-porter2](http://snowball.tartarus.org/algorithms/english/stemmer.html) stemming algorithm. The *porter2_stemmer* applies to all methods of the *textTinyPy* package.
    + On the other hand, *n-gram stemming* is *"language independent"* and supported by the **ngram_sequential** and **ngram_overlap** functions. The *n-gram stemming* applies to all methods except for the *Term_Matrix*, *transform_vec_docs* and *vocabulary_parser* methods of the corresponding *tokenizer*, *docs_matrix* and *utils* classes
        - [*ngram_overlap*](http://clef.isti.cnr.it/2007/working_notes/mcnameeCLEF2007.pdf) : The ngram_overlap stemming method is based on N-Gram Morphemes for Retrieval, Paul McNamee and James Mayfield
        - [*ngram_sequential*](https://arxiv.org/pdf/1312.4824.pdf) : The ngram_sequential stemming method is a modified version based on Generation, Implementation and Appraisal of an N-gram based Stemming Algorithm, B. P. Pande, Pawan Tamta, H. S. Dhami


<br><br>


### *utils* class

<br>

The following code chunks illustrate the *utils* of the package (besides the **read_characters()** and **read_rows()** which were used in the previous code chunks),

<br>

```py

#---------------------------------------
# cosine distance between word sentences
#---------------------------------------

s1 = 'sentence with two words'

s2 = 'sentence with three words'

sep = " "

init_utl = utils()

init_utl.cosine_distance(s1, s2, split_separator = sep)

```

```py

0.75

```


<br>

```R

#------------------------------------------------------------------------
# dice distance between two words (using n-grams -- the lower the better)
#------------------------------------------------------------------------

w1 = 'word_one'

w2 = 'word_two'

n = 2

init_utl.dice_distance(w1, w2, n_grams = n)


```


```py

0.2857143

```

<br>

```py

#---------------------------------------
# levenshtein distance between two words
#---------------------------------------

w1 = 'word_two'

w2 = 'word_one'

init_utl.levenshtein_distance(w1, w2)

```

```py

3.0

```

<br>

```py

#---------------------------------------------
# bytes converter (returns the size of a file)
#---------------------------------------------

PATH = "/planets.txt"

init_utl.bytes_converter(input_path_file = PATH, unit = "KB" )

```


```py

2.213867

```
<br>

```py

#---------------------------------------------------
# returns the utf-locale for the available languages
#---------------------------------------------------


init_utl.utf_locale(language = "english")

```

```py

"en.UTF-8"

```

<br>

```py

#-----------------
# text file parser
#-----------------

# The text file should have a structure (such as an xml-structure), so that 
# subsets can be extracted using the "start_query" and "end_query" parameters.
# (it works similarly to the big_text_parser() method, however for small to medium sized files)

# example input "example_file.xml" file :

<?xml version="1.0"?>
<sked>
  <version>2</version>
  <flight>
    <carrier>BA</carrier>
    <number>4001</number>
    <date>2011-07-21</date>
  </flight>
  <flight cancelled="true">
    <carrier>BA</carrier>
    <number>4002</number>
    <date>2011-07-21</date>
  </flight>
</sked>



```

<br>

```py

fp = init_utl.text_file_parser(input_path_file = "example_file.xml", 

                               output_path_file = "/output_folder/example_output_file.txt", 

                               start_query = '<number>', end_query = '</number>',

                               min_lines = 1, trimmed_line = False)

```


```py

"example_output_file.txt" :

4001
4002

```

<br>

```py

#------------------
# vocabulary parser
#------------------

# the 'vocabulary_parser' function extracts a vocabulary from a structured text (such as 
# an .xml file) and works in the exact same way as the 'big_tokenize_transform' class, 
# however for small to medium sized data files


pars_dat = init_utl.vocabulary_parser(input_path_file = '/folder/input_data.txt',

                                      start_query = 'start_word', end_query = 'end_word',
                                      
                                      vocabulary_path_file = '/folder/vocab.txt', 
                                      
                                      to_lower = True, split_string = True,
                                      
                                      remove_stopwords = True)

```


<br>

The last example chunks are part of the [documentation](https://mlampros.github.io/textTinyPy/_autosummary/utils.html#utils.utils.xml_parser_subroot_elements) and explain the **xml_parser_subroot_elements()** and **xml_parser_root_elements()** methods, which do xml-tree traversal using the boost library. The logic behind root-child-subchildren of an xml file is explained in [http://www.w3schools.com/xml/xml_tree.asp](http://www.w3schools.com/xml/xml_tree.asp),

<br>

```py

#----------------------------
# xml_parser_subroot_elements
#----------------------------


Example:


using the following structure as a 'FILE.xml'
---------------------------------------------

<mediawiki>
    <page>
        <title>AccessibleComputing</title>
        <redirect title="Computer accessibility"/>
        <revision>
            <id>631144794</id>
            <parentid>381202555</parentid>
            <timestamp>2014-10-26T04:50:23Z</timestamp>
        </revision>
    </page>
</mediawiki>



example to obtain a "subchild's element"   
----------------------------------------    

(here the xml_path equals to -->  "/root/child/subchild.element.sub-element")



utl = utils()                

res = utl.xml_parser_subroot_elements(input_path_file = "FILE.xml", 

                                      xml_path = "/mediawiki/page/revision.id", 
                                      
                                      empty_key = "")

```

```py

res                     # example output


array(['631144794'], 
      dtype='|S9')

```

<br>

```py

example to obtain a "subchild's attribute" ( by using the ".<xmlattr>." in the query )
------------------------------------------

the attribute in this .xml file is:     <redirect title="Computer accessibility"/>        


utl = utils()                

res = utl.xml_parser_subroot_elements(input_path_file = "FILE.xml", 

                                      xml_path = "/mediawiki/page/redirect.<xmlattr>.title")


```

```py

res                     # example output


array(['Computer accessibility'], 
      dtype='|S22')
      
```

<br>

xml file tree traversal for a root’s attributes using the boost library ( *repeated tree sturcture* ),

<br>


```py

#-------------------------
# xml_parser_root_elements
#-------------------------


Example:


using the following structure as a 'FILE.xml'
---------------------------------------------

<MultiMessage>
    <Message structID="1710" msgID="0" length="50">
        <structure type="AppHeader">
        </structure>
    </Message>
    <Message structID="27057" msgID="27266" length="315">
        <structure type="Container">
            <productID value="166"/>
            <publishTo value="xyz"/>
            <templateID value="97845"/>
        </structure>
    </Message>
</MultiMessage>
<MultiMessage>
    <Message structID="1710" msgID="0" length="50">
        <structure type="AppHeader">
        </structure>
    </Message>
    <Message structID="27057" msgID="27266" length="315">
        <structure type="Container">
            <productID value="166"/>
            <publishTo value="xyz"/>
            <templateID value="97845"/>
        </structure>
    </Message>
</MultiMessage>



example to get a "child's attributes" (use only the root-element of the xml file as parameter )
-------------------------------------


utl = utils()                

res = utl.xml_parser_root_elements(input_path_file = "FILE.xml", 

                                   xml_root = "MultiMessage", 
                                   
                                   output_path_file = "")

```


```py

res                     # example output


  child_keys child_values
0   structID         1710
1      msgID            0
2     length           50
3   structID        27057
4      msgID        27266
5     length          315

[6 rows x 2 columns]

```


<br>

An updated version of the textTinyPy package can be found in my [Github repository](https://github.com/mlampros/textTinyPy) and to report bugs/issues please use the following link, [https://github.com/mlampros/textTinyPy/issues](https://github.com/mlampros/textTinyPy/issues).


<br>

