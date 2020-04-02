# PSVO: Particle Smoothing Variational Objectives

This code provides a reference implementation of the Smoothing Variational Objectives (SVO) algorithms described in the publications: 


* [Variational Objectives for Markovian Dynamics with Backwards Simulation](https://arxiv.org/abs/1909.09734). \
  Moretti, A.\*, Wang, Z.\*, Wu, L.\*, Drori, I., Pe'er, I. \
  European Conference on Artificial Intelligence, 2020.

* [Particle Smoothing Variational Objectives](https://arxiv.org/abs/1909.09734). \
  Moretti, A.\*, Wang, Z.\*, Wu, L.\*, Drori, I., Pe'er, I. \
  arXiv preprint, arXiv:1909.097342019.

* [Smoothing Nonlinear Variational Objectives with Sequential Monte Carlo](https://openreview.net/pdf?id=HJg24U8tuE). \
  Moretti, A.\*, Wang, Z.\*, Wu, L., Pe'er, I. \
  ICLR Workshop on Deep Generative Models for Highly Structured Data, 2019.

SVO is written as an abstract class that reduces to two related variational inference methods for time series. As a reference, the AESMC and IWAE algorithms are implemented from the following publications:

* [Auto-Encoding Sequential Monte Carlo](https://arxiv.org/abs/1705.10306). \
  Le, T., Igl, M., Rainforth, T., Jin, T., Wood, F. \
  International Conference on Learning Representations, 2018.

* [Importance Weighted Autoencoders](https://arxiv.org/abs/1509.00519). \
  Burda, Y., Grosse, R., Salakhutidinov, R. \
  International Conference on Learning Representations, 2016.


## Installation

The code is written in Python 3.6. The following dependencies are required:

* Tensorflow
* seaborn
* numpy
* scipy 
* matplotlib

To check out, run `git@github.com:amoretti86/psvo.git`


## Usage

Running `python runner_flags.py` will find a two dimensional representation of the Fitzhugh-Nagumo dynamical system from one dimensional observations. The following figure provides the original dynamical system and trajectories along with the resulting inferred dynamics and trajectories from SVO. 

## Demo

| Original | Inferred |
|:--------------------------:|:--------------------------:|
|![fhn](https://github.com/amoretti86/PSVO/blob/master/data/fhn/fhn.png)|![fit](https://github.com/amoretti86/PSVO/blob/master/data/fhn/fit.png)|


