import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb

class NF:
	def __init__(self,
				 n_layers,
				 event_size,
				 hidden_layers = [512, 512],
				 flow_type = "IAF",
				 name = "NF"):
		self.event_size = event_size
		self.flow_type = flow_type
		self.name = name
		self.bijector = self.init_bijectors(n_layers, hidden_layers)

	@staticmethod
	def init_once(x, name):
		return tf.get_variable(name, dtype = tf.int32, initializer=x, trainable=False)

	def init_bijectors(self, n_layers, hidden_layers):
		with tf.variable_scope(self.name):
			bijectors = []
			for i in range(n_layers):
				if self.flow_type == "MAF":
					bijectors.append(tfb.MaskedAutoregressiveFlow(
						shift_and_log_scale_fn=tfb.masked_autoregressive_default_template(
							hidden_layers=hidden_layers,
							name = "MAF_template_{}".format(i)),
						name = "MAF_{}".format(i)))
				elif self.flow_type == "IAF":
					bijectors.append(
						tfb.Invert(
							tfb.MaskedAutoregressiveFlow(
								shift_and_log_scale_fn=tfb.masked_autoregressive_default_template(
									hidden_layers=hidden_layers,
									name = "MAF_template_{}".format(i))
								),
							name = "IAF_{}".format(i)
						)
					)
				bijectors.append(tfb.Permute(permutation=self.init_once(
					np.random.permutation(self.event_size).astype("int32"),
					name="permutation_{}".format(i))))

			flow_bijector = tfb.Chain(list(reversed(bijectors[:-1])))

			return flow_bijector

	def transform(self, base_dist, name=None):
		dist = tfd.TransformedDistribution(
			distribution=base_dist,
			bijector=self.bijector,
			name=name)

		return dist

	def sample_and_log_prob(self, base_dist, sample_size=(), name=None):
		dist = self.transform(base_dist, name)
		sample = dist.sample(sample_size)
		log_prob = dist.log_prob(sample)
		return dist, sample, log_prob

if __name__ == "__main__":
	Dx = 2
	batch_size = 5
	n_particles = 10
	flow = NF(2, Dx)
	mvn = tfd.MultivariateNormalDiag(tf.zeros((n_particles, batch_size, Dx)), name="mvn")
	dist, sample, log_prob = flow.sample_and_log_prob(mvn, name="trans_mvn")
	print(dist)
	print(sample)
	print(log_prob)
	