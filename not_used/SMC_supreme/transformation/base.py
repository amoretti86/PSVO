from abc import ABC, abstractmethod

# base class for transformation
class transformation(object):
	def __init__(self, params = None):
		self.params = params

	@abstractmethod
	def transform(self, X_prev):
		pass
		