import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

w = tf.Variable(np.array([1,2,3,4,5]), dtype = tf.float32, name = 'w1')
# c = tfd.Categorical(probs = w1)
# idx1 = tf.stop_gradient(c1.sample(5))
# sum1 = tf.reduce_sum(w1)
# w_1 = 100 * tf.gather(w1, idx1)
# sum1 += tf.reduce_sum(w_1)
# avg = tf.reduce_mean(w)
# w = w*w
w_ = tf.identity(w)
w = w*2



# train_op1 = tf.train.AdamOptimizer(0.0001).minimize(sum1)
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)

print(w.eval())
print(w_.eval())