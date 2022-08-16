---
layout: post
title: Functionality of the fastGLCM R package
tags: [R, package, R-bloggers]
comments: true
---

This blog post is a slight modification of the R package Vignette and
shows how to use the Rcpp Armadillo version of the
[fastGLCM](https://github.com/mlampros/fastGLCM) R package. The
**fastGLCM** R package is an *RcppArmadillo* implementation of the
Python Code for *Fast Gray-Level Co-Occurrence Matrix by numpy*,

-   [Github repository of the Python
    code](https://github.com/tzm030329/GLCM)
-   “Artifact-Free Thin Cloud Removal Using Gans” by Toizumi, Takahiro
    and Zini, Simone and Sagi, Kazutoshi and Kaneko, Eiji and Tsukada,
    Masato and Schettini, Raimondo in IEEE International Conference on
    Image Processing (ICIP), pp. 3596-3600, 2019,
    <https://doi.org/10.1109/ICIP.2019.8803652>

The python version works similarly and is included as an R6 class (see
the [documentation of
*fastglcm*](https://mlampros.github.io/fastGLCM/reference/fastglcm.html)).
However, it requires a python configuration in the user’s operating
system and additionally the installation of the
[reticulate](https://github.com/rstudio/reticulate) R package. <br>

For the theoretical background of the *Gray-Level Co-Occurrence Matrix
Textures* the user can consult an [existing Tutorial of the University
of
Calgary](https://prism.ucalgary.ca/bitstream/handle/1880/51900/texture%20tutorial%20v%203_0%20180206.pdf).

<br>

## Sample Satellite Imagery

<br>

The *fastGLCM* R package includes an *ALOS-3 simulation image* from JAXA
(Japan Aerospace Exploration Agency) in compressed format (.zip) around
Joso City, Ibaraki Prefecture from September 11, 2015, that will be used
in this blog-post for illustration purposes.

Both *fastGLCM* versions of the R package take a 2-dimensional object as
input (numeric matrix) and it is required that the range of pixel values
are between 0 and 255,

``` r
require(fastGLCM)
#> Loading required package: fastGLCM
require(OpenImageR)
#> Loading required package: OpenImageR
require(utils)

temp_dir = tempdir(check = FALSE)
# temp_dir

zip_file = system.file('images', 'JAXA_Joso-City2_PAN.tif.zip', package = "fastGLCM")
utils::unzip(zip_file, exdir = temp_dir)
path_extracted = file.path(temp_dir, 'JAXA_Joso-City2_PAN.tif')

im = readImage(path = path_extracted)
dim(im)
#> [1] 1555 1414
```

<br>

``` r
imageShow(im)
```

![Alt Text](/images/fastGLCM_images/input_image.png)

<br>

To decrease the computation time the initial width and height will be
reduced to 500,

``` r
#....................................................
# the pixel values will be adjusted between 0 and 255
#....................................................

im = resizeImage(im, 500, 500, 'nearest')
im = OpenImageR::norm_matrix_range(im, 0, 255)

#---------------------------------
# computation of all GLCM features
#---------------------------------

methods = c('mean',
            'std',
            'contrast',
            'dissimilarity',
            'homogeneity',
            'ASM',
            'energy',
            'max',
            'entropy')

res_glcm = fastGLCM_Rcpp(data = im,
                         methods = methods,
                         levels = 8,
                         kernel_size = 5,
                         distance = 1.0,
                         angle = 0.0,
                         threads = 1,
                         verbose = TRUE)
#> Elapsed time: 0 hours and 0 minutes and 1 seconds.

if (file.exists(path_extracted)) file.remove(path_extracted)
#> [1] TRUE

str(res_glcm)
#> List of 9
#>  $ mean         : num [1:500, 1:500] 0.578 0.766 0.953 0.938 0.938 ...
#>  $ std          : num [1:500, 1:500] 28.3 40 51.8 59.5 59.5 ...
#>  $ contrast     : num [1:500, 1:500] 2 2 2 0 0 1 2 4 4 4 ...
#>  $ dissimilarity: num [1:500, 1:500] 2 2 2 0 0 1 2 4 4 4 ...
#>  $ homogeneity  : num [1:500, 1:500] 8 11 14 15 15 14.5 14 13 13 13 ...
#>  $ ASM          : num [1:500, 1:500] 51 102 171 225 225 147 107 73 73 73 ...
#>  $ energy       : num [1:500, 1:500] 7.14 10.1 13.08 15 15 ...
#>  $ max          : num [1:500, 1:500] 7 10 13 15 15 12 10 8 8 8 ...
#>  $ entropy      : num [1:500, 1:500] 8.59 8.49 8.42 8.07 8.07 ...
```

<br>

The output matrices based on the selected methods (*mean*, *std*,
*contrast*, *dissimilarity*, *homogeneity*, *ASM*, *energy*, *max*,
*entropy*) can be visualized in a multi-plot,

``` r
plot_multi_images(list_images = res_glcm,
                  par_ROWS = 2,
                  par_COLS = 5,
                  titles = methods)
```

<img src="/images/fastGLCM_images/multiplot.png" width="900" height="560">

<br>

**Credits:**

-   The [ALOS-3 simulation
    image](https://www.eorc.jaxa.jp/ALOS/en/alos-3/datause/a3_simulation_e.htm)
    is based on the sample product provided by JAXA. Please, read the
    [terms of use for this sample
    product](https://earth.jaxa.jp/en/data/policy/)

<br>

### Package Installation & Citation:

<br>

To install the package from CRAN use,

``` r
install.packages("fastGLCM")
```

<br>

and to download the [latest version of the package from
Github](https://github.com/mlampros/fastGLCM),

``` r
remotes::install_github('mlampros/fastGLCM')
```

<br>

If you use the **fastGLCM** R package in your paper or research please
cite both **fastGLCM** and the **original articles / software**
`https://cran.r-project.org/web/packages/fastGLCM/citation.html`:

<br>

``` r
@Manual{,
  title = {fastGLCM: Fast Gray Level Co-occurrence Matrix computation (GLCM) using R},
  author = {Lampros Mouselimis},
  year = {2022},
  note = {R package version 1.0.0},
  url = {https://CRAN.R-project.org/package=fastGLCM},
}
```

<br>
