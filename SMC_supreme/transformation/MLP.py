import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.initializers import Constant

from transformation.base import transformation


class MLP_transformation(transformation):
    def __init__(self, Dhs, Dout,
                 use_residual=False,
                 output_cov=False,
                 diag_cov=False,
                 dropout_rate=0.2,
                 name="MLP_transformation"):
        self.Dhs = Dhs
        self.Dout = Dout

        self.use_residual = use_residual
        self.output_cov = output_cov
        self.diag_cov = diag_cov
        self.dropout_rate = dropout_rate

        self.name = name
        self.init_FFN()

    def init_FFN(self):
        with tf.variable_scope(self.name):
            self.hidden_layers = []
            # self.dropout_layers = []
            for i, Dh in enumerate(self.Dhs):
                self.hidden_layers.append(
                    Dense(Dh,
                          activation="relu",
                          kernel_initializer="he_uniform",
                          name="hidden_{}".format(i))
                )
                # self.dropout_layers.append(
                #     Dropout(rate=self.dropout_rate,
                #             name="dropout_{}".format(i))
                # )

            self.mu_layer = Dense(self.Dout,
                                  activation="linear",
                                  kernel_initializer="he_uniform",
                                  name="mu_layer")

            if self.output_cov:
                sigma_dim = self.Dout if self.diag_cov else self.Dout**2
                self.sigma_layer = Dense(sigma_dim,
                                         activation="linear",
                                         kernel_initializer="he_uniform",
                                         bias_initializer=Constant(1.0),
                                         name="sigma_layer")

    def transform(self, Input):
        with tf.variable_scope(self.name):
            hidden = tf.identity(Input)
            # for hidden_layer, dropout_layer in zip(self.hidden_layers, self.dropout_layers):
            #     hidden = hidden_layer(hidden)
            #     hidden = dropout_layer(hidden)
            for hidden_layer in self.hidden_layers:
                hidden = hidden_layer(hidden)

            mu = self.mu_layer(hidden)

            cov = None
            if self.output_cov:
                cov = self.sigma_layer(hidden)
                if self.diag_cov:
                    cov = tf.exp(cov) + 1e-6  # to resolve numerical issues
                else:
                    batch_size = hidden.shape.as_list()[:-1]
                    cov = tf.reshape(cov, batch_size + [self.Dout, self.Dout]) + 1e-6
                    cov = tf.matmul(cov, cov, transpose_b=True)

        return mu, cov

    def get_variables(self):
        res_dict = {}

        layers = self.hidden_layers + [self.mu_layer]
        if self.output_cov:
            layers += [self.sigma_layer]

        for layer in layers:
            variable1, variable2 = layer.variables
            if len(variable1.shape.as_list()) == 2:
                weights, bias = variable1, variable2
            else:
                weights, bias = variable2, variable1
            res_dict[layer.name + "/weights"] = weights
            res_dict[layer.name + "/bias"] = bias

        return res_dict
