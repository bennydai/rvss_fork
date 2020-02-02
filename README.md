# RVSS Workshop 2020 Compute Environment

## Overview
The workshop task will be implemented in Python. The provided software is expected to work across platform but has only been tested in Linux. If you have difficulties installing the required software we will have laptops available to borrow. We recommend installing the required software in a Conda environment to avoid interference with existing installations. Conda handles dependencies seamlessly and makes it easy to set up different environments with different versions of libraries. And if an environment is ruined beyond repair, you can just remove it and start over with a clean one.

## Installing Conda
* Install Conda from [here][Conda].
* Select the right Python 3 version appropriate to your machine (e.g Windows/Mac/Linux)
* If asked for advanced options.  Do not select “add anaconda to the system path environment” or “register anaconda as the system Python 3.7”. 
* For Windows: install the .exe. Once installed on the "start" menu open the "anaconda prompt"
* For Linux/Mac:  
```{p}
bash Miniconda3-*.sh
```

## Setting up a Conda Environment

* To create a new environment:
```{p}
conda create --name rvss2020 python=3
```
* To activate the environment:
```{p}
conda activate rvss2020
```
* When you are finished in an environment you can leave by:
```{p}
conda deactivate rvss2020
```
* To see your environments:
```{p}
conda env list
```
* To remove an environment:
```{p}
conda remove --name FAILED_ENVIRONMENT --all
```
## Installing Required Software
In a terminal use the following commands to install the required packages.
**Note: Install packages needed for the workshop inside the environment by activating it first!**
```{p}
conda install opencv
conda install pytorch=0.4.1 torchvision -c pytorch
conda install scipy numpy
conda install -c conda-forge matplotlib
conda install -c anaconda requests
pip install getch
pip install Pillow==4.1.1
pip install PyYAML
```




[conda]: https://conda.io/miniconda.html "Conda"

