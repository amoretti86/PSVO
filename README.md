# PSVO: Particle Smoothing Variational Objectives

This code provides a reference implementation of the Smoothing Variational Objectives (SVO) alorithm described in the publication: 

* Moretti, A., Wang, Z., Wu, L., Pe'er, I. [Smoothing Nonlinear Variational Objectives with Sequential Monte Carlo](https://openreview.net/pdf?id=HJg24U8tuE). ICLR Workshops, 2019.

SVO is written as an abstract class that reduces to two related methods. As a reference, the AESMC and IWAE algorithms are implemented from the following publications:

* Burda, Y., Grosse, R., Salakhutidinov, R. [Importance Weighted Autoencoders](https://arxiv.org/abs/1509.00519). ICLR, 2016.

* Le, T., Igl, M., Rainforth, T., Jin, T., Wood, F. [Auto-Encoding Sequential Monte Carlo](https://arxiv.org/abs/1705.10306). ICLR, 2018.


## Installation

The code is written in Python 3.6. Tensorflow, seaborn, numpy, scipy and matplotlib are expected. To check out, run <b>git@github.com:amoretti86/psvo.git</b>


## Usage

Running python runner_flags.py will find a two dimensional representation of the Fitzhugh-Nagumo dynamical system from one dimensional observations. The following figure provides the original dynamical system and trajectories along with the resulting inferred dynamics and trajectories from SVO. 


## Demo

| Original | Inferred |
|-----------|----------|
| <img src="https://github.com/amoretti86/PSVO/tree/master/notebooks/figures/fhn.png" width="100%"/> 
| <img src="https://github.com/amoretti86/PSVO/tree/master/notebooks/figures/fit.png" width="100%"/> |
