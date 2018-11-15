import numpy as np
import math
from sklearn.utils import shuffle

import tensorflow as tf

class trainer:
	def __init__(Dx, Dy, 
				 n_particles, time, 
				 batch_size, lr, epoch,
				 n_train, n_test,
				 print_freq, save_freq)