# PSVO: Particle Smoothing Variational Objectives

This code provides a reference implementation of the SVO alorithm described in the publication: 

A Moretti*, Z Wang*, L Wu, I, Pe'er. [Smoothing Nonlinear Variational Objectives with Sequential Monte Carlo](https://openreview.net/pdf?id=HJg24U8tuE). ICLR Workshops, 2019.

PSVO reduces to several simpler algorithms. As a reference, the AESMC and IWAE algorithms are implemented from the following publications:

Y Burda, R Grosse, R Salakhutidinov. [Importance Weighted Autoencoders](https://arxiv.org/abs/1509.00519). ICLR, 2016.

T Le, M Igl, T Rainforth, T Jin, F Wood (2018). [Auto-Encoding Sequential Monte Carlo](https://arxiv.org/abs/1705.10306). ICLR, 2018.


## Installation

The code is written in Python 3.6. Tensorflow 1.12, seaborn, numpy, scipy and matplotlib are expected. To check out, run git@github.com:amoretti86/psvo.git


## Usage

Running python runner_flags.py will find a two dimensional representation of the Fitzhugh-Nagumo dynamical system from one dimensional observations. The following figure provides the original dynamical system and trajectories along with the resulting inferred dynamics and trajectories from SVO. 


## Demo

| Original | Inferred |
|-----------|----------|
|<img src="https://github.com/amoretti86/VISMC/blob/developments/notebooks/fhn.png" width="100%" /> | <img src="https://github.com/amoretti86/VISMC/blob/developments/notebooks/fit.gif" width="50%" /> |
