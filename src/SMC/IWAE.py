import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from SMC.SVO import SVO


class IWAE(SVO):
    def __init__(self, model, FLAGS, name="log_ZSMC"):
        SVO.__init__(self, model, FLAGS, name)
        self.resample_particles = False
        self.smooth_obs = False
