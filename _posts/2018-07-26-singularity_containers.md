---
layout: post
title: Singularity as a software distribution / deployment tool
tags: [R, Python, setups, Singularity, R-bloggers]
comments: true
---


In this blog post, I'll explain how someone can take advantage of [Singularity](https://github.com/singularityware/singularity) to make R or Python packages available as an image file to users. This is a necessity if the specific R or Python package is difficult to install across different operating systems making that way the installation process cumbersome. Lately, I've utilized the [reticulate](https://github.com/rstudio/reticulate) package in R (it provides an interface between R and Python) and I realized from first hand how difficult it is, in some cases, to install R and Python packages and make them work nicely together in the same operating system. This blog post by no means presents the potential of Singularity or containerization tools, such as [docker](https://github.com/docker), but it's mainly restricted to package distribution / deployment. 

<br>

Singularity can be installed on all 3 operating systems (Linux, Macintosh, Windows), however the current status (as of July 2018) is that on Macintosh and Windows the user has to setup [Vagrant](https://www.vagrantup.com/), and [run Singularity](https://www.sylabs.io/guides/2.5.1/user-guide.pdf#page=21) from there ([this might change in the near future](https://groups.google.com/a/lbl.gov/forum/#!msg/singularity/7d9F16DCG8Y/iXo_wRxtCwAJ)). 

<br><br>


### Singularity on Linux

<br>

In the following lines I'll make use of an Ubuntu cloud instance (the same steps can be accomplished on an Ubuntu Desktop with some exceptions) to explain how someone can download Singularity image files and run those images on Rstudio server (in case of R) or a Jupyter Notebook (in case of Python). I utilize Amazon Web Services ([AWS](https://aws.amazon.com/)) and especially an *Ubuntu server 16.04* using a *t2.micro* instance (1GB memory, 1 core), however, someone can follow the same procedure on [Azure](https://azure.microsoft.com/en-us/) or [Google Cloud](https://cloud.google.com/) (at least of those two alternative cloud services I'm aware) as well. I'll skip the steps on how someone can set-up an Ubuntu cloud instance, as it's beyond the scope of this blog post (there are certainly many tutorials on the web for this purpose).

<br>

Assuming someone uses the command line console, the first thing to do is to install the [system requirements](https://github.com/mlampros/singularity_containers/blob/master/system_requirements.sh) (in case of an Ubuntu Desktop upgrading the system should be skipped),

<br>

```R

bash system_requirements.sh


```

<br>

Once the installation of the system requirements is finished the following folder should appear in the home directory,

<br>

```R

singularity


```

<br><br>

### R language Singularity image files

<br>

My [singularity_containers](https://github.com/mlampros/singularity_containers) Github repository contains R and Python [Singularity Recipes](https://github.com/singularityhub/singularityhub.github.io/wiki/Build-A-Container), which are used to build the corresponding containers. My Github repository is connected to my [singularity-hub](https://singularity-hub.org/collections/1321) account and once a change is triggered (for instance, a push to my repository) a new / updated container build will be created. An updated build - for instance for the [RGF package](https://github.com/mlampros/RGF) - can be pulled from singularity-hub in the following way,

<br>

```R

singularity pull --name RGF_r.simg shub://mlampros/singularity_containers:rgf_r


```

<br>

This code line will create the *RGF_r.simg* image file in the home directory. One should now make sure that port 8787 is not used by another service / application by using,

<br>

```R

sudo netstat -plnt | fgrep 8787


```

<br>

If this does not return something then one can proceed with, 

<br>

```R

singularity run RGF_r.simg


```

<br>

to run the image. If everything went ok and no errors occurred then by opening a second command line console and typing,


<br>

```R

sudo netstat -plnt | fgrep 8787


```

<br>

one should observe that port 8787 is opened, 

<br>

```R

tcp        0      0 0.0.0.0:8787            0.0.0.0:*               LISTEN      23062/rserver


```

<br>

The final step is to open a web-browser (chrome, firefox etc.) and give,

<br>

* http://Public DNS (IPv4):8787  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  ( *where "Public DNS (IPv4)" is specific to the Cloud instance you launched* )

or 

* http://0.0.0.0:8787  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  ( *in case that someone uses Singularity locally* )


<br>

to launch the Rstudio-server and use the RGF package pre-installed with all requirements included (to stop the service use *CTRL + C* from the command line). I used RGF as an example here because for me personally, it was somehow cumbersome to install on my windows machine.


The same applies to the other two *R* singularity recipe files included in my singularity-hub account, i.e. *mlampros/singularity_containers:nmslib_r* and *mlampros/singularity_containers:fuzzywuzzy_r*.

<br>

### Python language Singularity image files

<br>

The Python Singularity Recipe files which are also included in the same Github repository utilize port 8888 and follow a similar logic with the R files. The only difference is that when a user **runs** the image the **sudo** command is required (otherwise it will raise a permission error),

<br>

```R

singularity pull --name RGF_py.simg shub://mlampros/singularity_containers:rgf_python

sudo singularity run RGF_py.simg


```

<br>

The latter command will produce the following (example) output,

<br>

```R

The web-browser runs on localhost:8888
[I 09:56:03.427 NotebookApp] Writing notebook server cookie secret to /root/.local/share/jupyter/runtime/notebook_cookie_secret
[W 09:56:03.779 NotebookApp] WARNING: The notebook server is listening on all IP addresses and not using encryption. This is not recommended.
[I 09:56:03.789 NotebookApp] Serving notebooks from local directory: /root
[I 09:56:03.790 NotebookApp] The Jupyter Notebook is running at:
[I 09:56:03.790 NotebookApp] http://(ip-172-31-21-76 or 127.0.0.1):8888/?token=1fc90f01247498dac8d24ac918fe8da57fa46ee9e98eea4f
[I 09:56:03.790 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[C 09:56:03.790 NotebookApp] 
    
    Copy/paste this URL into your browser when you connect for the first time,
    to login with a token:
        http://(ip-172-31-21-76 or 127.0.0.1):8888/?token=1fc90f01247498dac8d24ac918fe8da57fa46ee9e98eea4f

.......

```

<br>

In the same way as before the user should open a web-browser and give either,


<br>

* http://Public DNS (IPv4):8888  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  ( *where "Public DNS (IPv4)" is specific to the Cloud instance you launched* )

or 

* http://127.0.0.1:8888  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  ( *in case that someone uses Singularity locally* )


<br>


When someone connects for the first time to the Jupyter notebook then he / she has to give the output token as the password. For instance, based on the previous example output the token password would be *1fc90f01247498dac8d24ac918fe8da57fa46ee9e98eea4f*. 

I also included an [.ipynb](https://github.com/mlampros/singularity_containers/blob/master/Python/regression_RGF.ipynb) file which can be loaded to the Jupyter notebook to test the [rgf_python](https://github.com/RGF-team/rgf/tree/master/python-package) package.

The same applies to the other two *Python* singularity recipe files included in my singularity-hub account, i.e. *mlampros/singularity_containers:nmslib_python* and *mlampros/singularity_containers:fuzzywuzzy_python*. 

<br>

### Final words

<br>

If someone intends to add authentication to the Singularity recipe files then valuable resources can be found in the [https://github.com/nickjer/singularity-rstudio](https://github.com/nickjer/singularity-rstudio) Github repository, on which my Rstudio-server recipes heavily are based.  

<br>

An updated version of *singularity_containers* can be found in my [Github repository](https://github.com/mlampros/singularity_containers) and to report bugs &nbsp; / &nbsp; issues please use the following link, [https://github.com/mlampros/singularity_containers/issues](https://github.com/mlampros/singularity_containers/issues).

<br><br>

**References** :

* https://github.com/singularityhub
* https://vsoch.github.io/
* https://github.com/nickjer/singularity-rstudio
* https://github.com/nickjer/singularity-r
* https://bwlewis.github.io/r-and-singularity/

<br><br>
