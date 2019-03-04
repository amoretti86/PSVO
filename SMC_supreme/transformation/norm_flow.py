"""Class for implementation of major transformations."""

import numpy as np
import tensorflow as tf

class Transform(object):

    def __init__(self, in_dim, out_dim, gov_param=None, initial_value=None,
            name=None):
        """Sets up the universal properties of any transformation function.

        Governing parameters of the transformation is set in the constructor.

        params:
        -------
        in_dim: int
            dimensionality of the input code/variable.
        out_dim: int
            dimensionality of the output code/variable.
        gov_param: tf.Tensor
            In case that parameters of the transformation are governed by
            another tensor.
        initial_value: numpy.ndarray
            Initial value of the transformation variable.
        """
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.initial_value = initial_value
        if gov_param is not None and not isinstance(gov_param, tf.Tensor) and\
                not isinstance(gov_param, tf.Variable):
            raise ValueError(
                    'Governing parameters of transformation should be Tensor.')
        self.var = gov_param
        # Check that parameters are tensors.
        # The following variable have to be set in the constructor of each
        # sub-class. Param shape is the correct shape of the parameters of the
        # Transformation. This has to be a tuple.
        self.param_shape = None
        self.name = name

    def initializer(self):
        """Default initializer of the transformation class."""
        self.var = tf.Variable(np.random.normal(0, 1, self.param_shape))

    def check_param_shape(self):
        """Checks the shape of the governing parameters or init values.

        It also initializes the variables (i.e. parameters) if necessary.
        """
        if self.var is not None:
            if not self.var.shape == self.param_shape:
                raise ValueError("gov_param tensor's shape must be {}".format(
                    self.param_shape))
        elif self.initial_value is None:
            self.initializer()
        else:
            if not self.initial_value.shape == self.param_shape:
                raise ValueError("initial_value's shape must be {}.".format(
                    self.param_shape))
            self.var = tf.Variable(self.initial_value)

    def check_input_shape(self, x):
        """Checks whether the input has valid shape."""
        if not x.shape[-1] == self.in_dim:
            raise ValueError(
                    "Input must have dimension {}.".format(self.in_dim))

    def broadcast_operator(self, x):
        """Input of higher dimensions the operation will give same shape."""
        # Reshape the input array into 2 dimensions.
        input_ = x
        n_tot = 1
        for dim in x.shape[:-1]:
            n_tot *= dim.value
        input_ = tf.reshape(x, [n_tot, self.in_dim])
        output_ = tf.reshape(
                self.operator(input_), x.shape[:-1].as_list() + [self.out_dim])
        return output_

    def get_in_dim(self):
        return self.in_dim

    def get_out_dim(self):
        return self.out_dim

    def get_transformation_parameters(self):
        return self.var

    def get_name(self):
        return self.name

    def get_regularizer(self, scale=1.):
        """Computes the regularization of the parameter to be added to loss.

        returns:
        --------
        tensorflow.Tensor containing the regualization of the variables.
        """
        pass

    def operator(self, x):
        """Gives the tensorflow operation for transforming a tensor."""
        pass


class LinearTransform(Transform):

    def __init__(self, in_dim, out_dim, initial_value=None, gov_param=None,
            non_linearity=None, name=None):
        """Sets up the linear transformation and bias variables.

        params:
        -------
        non_linearity: tf.Operation
            Type of non-linearity to be used for the output. In effect, turning
            LinearTransform to a fully connected layer with non-linear
            activations. E.g. tf.nn.relu, tf.nn.sigmoid, tf.nn.tanh, etc. If
            None, the output is linear.
        """
        super(LinearTransform, self).__init__(
                in_dim, out_dim, gov_param=gov_param,
                initial_value=initial_value, name=name)
        # Make sure initialization and parameters are correct.
        self.param_shape = (self.in_dim + 1, out_dim)
        self.check_param_shape()

        # Partitioning the variable into respective variables of a linear
        # trnsformation.
        self.lin_trans = self.var[:-1]
        self.bias = self.var[-1]
        self.non_linearity = non_linearity

    def initializer(self):
        """Overriden function to do Xavier initialization."""
        self.var = tf.Variable(np.random.normal(
                0, np.sqrt(1. / self.in_dim), self.param_shape))

    def get_regularizer(self, scale=1.):
        return scale * tf.reduce_sum(tf.reduce_sum(tf.square(self.lin_trans)))

    def operator(self, x):
        if len(x.shape) > 2:
            return self.broadcast_operator(x)
        self.check_input_shape(x)
        t_matrix = self.lin_trans
        bias = self.bias
        linear_output = tf.matmul(x, t_matrix) + bias
        if self.non_linearity is not None:
            return self.non_linearity(linear_output)
        return linear_output 


class MultiLayerPerceptron(Transform):

    def __init__(self, in_dim, out_dim, hidden_units,
            activation=tf.nn.relu, output_activation=None, name=None):
        """
        Sets up the layers of the MLP transformation.

        params:
        -------
        hidden_units: list of int
            Number of hidden units per hidden layer respectively.
        activation: tf.Operation
            Activation of each hidden layer in the network.
        output_activation: tf.Operation
            Output layer's non-linearity function.
        """
        super(MultiLayerPerceptron, self).__init__(
                in_dim=in_dim, out_dim=out_dim, name=name)

        self.activation = activation
        self.out_activation = output_activation
        self.hidden_units = hidden_units
        # List that will containt the individual transformation for each layer.
        self.layers = []
        activation = self.activation
        for i, n_units in enumerate(hidden_units + [self.out_dim]):
            # Apply output layers non-linearity for if it is the last layer.
            if i == len(hidden_units):
                activation = self.out_activation
            layer_t = LinearTransform(
                    in_dim=in_dim, out_dim=n_units,
                    non_linearity=activation)
            self.layers.append(layer_t)
            in_dim = n_units

    def get_regularizer(self, scale=1.):
        """Regularizer for the weights of the multilayer perceptron."""
        sum_all = 0.
        for layer in self.layers:
            sum_all += layer.get_regularizer()
        return sum_all

    def get_transformation_parameters(self):
        """Returns the list of variables of the layers of the MLP.

        returns:
        --------
        list of tensorflow.Tensor
        """
        if self.var is None:
            self.var = []
            for layer in self.layers:
                self.var.append(layer.var)
        return self.var

    def operator(self, x):
        output = x
        for layer in self.layers:
            output = layer.operator(output)
        return output


class NormalizingFlow(Transform):

    def __init__(self, dim, gov_param=None, initial_value=None, name=None):
        """Sets up the universal properties of any transformation function.

        Governing parameters of the transformation is set in the constructor.

        params:
        -------
        dim: int
            dimensionality of the input code/variable and output variable.
        gov_param: tf.Tensor
            In case that parameters of the transformation are governed by
            another tensor.
        initial_value: numpy.ndarray
            Initial value of the transformation variable.
        """
        super(NormalizingFlow, self).__init__(in_dim=dim, out_dim=dim,
                gov_param=gov_param, initial_value=initial_value, name=name)
        self.dim = dim

    def log_det_jacobian(self, x):
        """Given x the log determinent Jacobian of the matrix.

        returns:
        --------
        tf.Tensor for log-det-jacobian of the transformation given x.
        """
        pass

    @staticmethod
    def get_param_shape(**kwargs):
        """Gets the shape of the governing parameters of the transform."""
        raise NotImplementedError(
                "Abstract method that each sub-class has to implement.")


class PlanarFlow(NormalizingFlow):

    def __init__(self, dim, n_flow=1, non_linearity=tf.tanh, gov_param=None,
            initial_value=None, enforce_inverse=True, name=None):
        """Sets up the universal properties of any transformation function.

        Governing parameters of the transformation is set in the constructor.

        params:
        -------
        dim: int
            dimensionality of the input code/variable and output variable.
        gov_param: tf.Tensor
            In case that parameters of the transformation are governed by
            another tensor.
        initial_value: numpy.ndarray
            Initial value of the transformation variable.
        enforce_inverse: bool
            If true, the parameters are changed slightly to guarantee
            invertibility.
        """
        super(PlanarFlow, self).__init__(dim=dim, gov_param=gov_param,
                initial_value=initial_value, name=name)
        # Set the rest of the attributes.
        self.non_linearity = non_linearity
        # Make sure the shape of the parameters is correct.
        self.n_flow = n_flow
        self.param_shape = PlanarFlow.get_param_shape(dim=dim, n_flow=n_flow)
        self.check_param_shape()

        # Partition the variable into variables of the planar flow.
        self.w = tf.slice(self.var, [0, 0], [-1, self.dim])
        self.u = tf.slice(self.var, [0, self.dim], [-1, self.dim])
        self.b = tf.slice(self.var, [0, 2 * self.dim], [-1, 1])
        # Guarantee invertibility of the forward transform.
        self.u_bar = self.u
        if enforce_inverse:
            self.enforce_invertiblity()

        # Tensor map for keeping track of computation redundancy.
        # If an operation on a tensor has been done before, do not redo it.
        self.tensor_map = {}

    def enforce_invertiblity(self):
        """Guarantee that planar flow does not have 0 determinant Jacobian."""
        if self.non_linearity is tf.tanh:
            dot = tf.reduce_sum(
                    (self.u * self.w), axis=1, keepdims=True)
            scalar = - 1 + tf.nn.softplus(dot) - dot
            norm_squared = tf.reduce_sum(
                    self.w * self.w, axis=1, keepdims=True)
            comp = scalar * self.w / norm_squared
            self.u_bar = self.u + comp
        elif self.non_linearity is tf.nn.softplus:
            self.u_bar = self.u

    def non_linearity_derivative(self, x):
        """Operation for the derivative of the non linearity function."""
        if self.non_linearity is tf.tanh:
            return 1. - tf.tanh(x) * tf.tanh(x)
        elif self.non_linearity is tf.nn.softplus:
            return tf.nn.sigmoid(x)

    def inner_prod(self, x):
        """Computes the inner product part of the transformation."""
        if x in self.tensor_map:
            return self.tensor_map[x]
 
        result = tf.matmul(
                x, tf.expand_dims(self.w, axis=1),
                transpose_b=True) + tf.expand_dims(self.b, axis=1)

        self.tensor_map[x] = result
        return result

    def operator(self, x):
        """Given x applies the Planar flow transformation to the input.

        params:
        -------
        x: tf.Tensor
            Input tensor for which the transformation is computed.

        returns:
        --------
        tf.Tensor.
        """
        dial = self.inner_prod(x)
        result = x + tf.expand_dims(self.u_bar, axis=1) * self.non_linearity(dial)
        return result

    def log_det_jacobian(self, x):
        """Computes log-det-Jacobian for combination of inputs, flows.

        params:
        -------
        x: tensorflow.Tensor
            Input tensor for which the log-det-jacobian is computed.

        returns:
        --------
        tf.Tensor for log-det-jacobian of the transformation given x.
        """
        dial = self.inner_prod(x)

        psi = self.non_linearity_derivative(dial)
        det_jac = tf.matmul(
                tf.expand_dims(self.u_bar, axis=1),
                tf.expand_dims(self.w, axis=1), transpose_b=True) * psi
        result = tf.squeeze(tf.log(tf.abs(1 + det_jac)))

        return result

    @staticmethod
    def get_param_shape(**kwargs):
        """Gets the shape of the governing parameters of the transform."""
        dim, n_flow = kwargs["dim"], kwargs["n_flow"]
        return (n_flow, 2 * dim + 1)


class MultiLayerPlanarFlow(NormalizingFlow):

    def __init__(self, dim, num_layer, n_flow=1, non_linearity=tf.tanh,
            gov_param=None, initial_value=None, name=None):
        """Sets up the universal properties of any transformation function.

        Governing parameters of the transformation is set in the constructor.

        params:
        -------
        dim: int
            dimensionality of the input code/variable and output variable.
        num_layer: int
            Number of successive layers of palanr flow transformation.
        gov_param: tf.Tensor
            In case that parameters of the transformation are governed by
            another tensor.
        initial_value: numpy.ndarray
            Initial value of the transformation variable.
        enforce_inverse: bool
            If true, the parameters are changed slightly to guarantee
            invertibility.
        """
        super(MultiLayerPlanarFlow, self).__init__(dim=dim, gov_param=gov_param,
                initial_value=initial_value, name=name)
        # Set the rest of the attributes.
        self.non_linearity = non_linearity
        self.num_layer = num_layer
        self.n_flow = n_flow
        # Make sure the shape of the parameters is correct.
        #self.param_shape = (self.num_layer, self.num, 2 * self.dim + 1)
        self.param_shape = MultiLayerPlanarFlow.get_param_shape(
                num_layer=self.num_layer, dim=self.dim, n_flow=self.n_flow)
        self.check_param_shape()
        # Create a flow transform for every layer.
        self.layers = []
        for i in range(self.num_layer):
            layer_gov_param = self.var[i]
            self.layers.append(PlanarFlow(
                dim=self.dim, n_flow=self.n_flow, non_linearity=non_linearity,
                gov_param=layer_gov_param, name=name))

    def operator(self, x):
        """Given x applies the Planar flow transformation to the input.

        params:
        -------
        x: tf.Tensor

        """
        result = x
        for layer in self.layers:
            result = layer.operator(result)
        return result

    def log_det_jacobian(self, x):
        """Computes log-det-Jacobian for combination of inputs, flows.

        params:
        -------
        x: tensorflow.Tensor

        returns:
        --------
        tf.Tensor for log-det-jacobian of the transformation given x.
        """
        result = 0.
        # Tensor for intermediate transformation results.
        inter_trans = x
        for layer in self.layers:
            result += layer.log_det_jacobian(inter_trans)
            inter_trans = layer.operator(inter_trans)
        return result

    @staticmethod
    def get_param_shape(**kwargs):
        """Gets the shape of the governing parameters of the transform.

        Parameters:
        -----------
        **kwargs:
            Paramters of the constructor that must include the following:
                dim,
                num,
                num_layer
        """
        num_layer, dim = kwargs["num_layer"], kwargs["dim"]
        n_flow = kwargs["n_flow"]

        return (num_layer, n_flow, 2 * dim + 1)


class CondNormFlow(object):

    def __init__(self, in_dim, out_dim, num_layer, mlp_hidden_units,
            non_linearity=tf.tanh, reparam_base=True):
        """Initializes the conditional distribution.

        params:
        -------
        in_dim: int
            Dimensionality of input variable that the distribution is
            conditioned on.
        out_dim: int
            dimensionality of the output random variable.
        num_layer: int
            Number of layers for the normalizing flow.
        mlp_hidden_units: list of int
            Hidden units per layer of the MLP respectively.
        non_linearity: tf.Operator
            Non-linearity for the planar flow.
        reparam_base: boolean
            If True, the base distribution's mean and scale is conditioned
            on the input. Otherwise, isotropic gaussian is used as base
            noise distribution.
        """
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_nf_layer = num_layer
        self.nf_non_linearity = non_linearity
        self.reparam_base = reparam_base
        # MLP that indexes into the location and scale of the base distribution
        # of the noise that is transformed with the normalizing flow.
        if self.reparam_base:
            base_param_dim = 2 * self.out_dim
            self.base_param_mlp = MultiLayerPerceptron(
                    in_dim=self.in_dim, out_dim=base_param_dim,
                    hidden_units=mlp_hidden_units)
        # MLP that indexes into the parameters of the normalizing flow.
        nf_param_dim = (self.out_dim * 2 + 1) * self.num_nf_layer
        self.nf_param_mlp = MultiLayerPerceptron(
                in_dim=self.in_dim, out_dim=nf_param_dim,
                hidden_units=mlp_hidden_units)

    def sample_log_prob(self, input_tensor, sample_size, name=None):
        """Samples from the conditional distribtion conditioned on given input.

        params:
        -------
        sample_shape: int
            Number of samples from the conditional distributions.
        input_tensor: tf.Tensor
            Shape of the input should be (?, in_dim).
        """
        # Total number of parallel normalizing flows that are applied.
        n_flow = input_tensor.shape[0].value

        assert len(input_tensor.shape) == 2, "Input tensor not correct size"
        assert input_tensor.shape[1].value == self.in_dim, "Incorrect input dim."

        dtype = input_tensor.dtype
        loc = tf.zeros(self.out_dim, dtype=dtype)
        scale = tf.ones(self.out_dim, dtype=dtype)
        if self.reparam_base:
            # Get concatenated loc and scale from their corresponding MLP.
            loc_scale = self.base_param_mlp.operator(input_tensor)
            loc = tf.slice(loc_scale, [0, 0], [-1, self.out_dim])
            scale = tf.nn.softplus(
                    tf.slice(loc_scale, [0, self.out_dim], [-1, self.out_dim]))

        base = tf.contrib.distributions.MultivariateNormalDiag(
                loc=loc, scale_diag=scale)

        if self.reparam_base:
            base_sample = base.sample(sample_size)
            base_log_prob = base.log_prob(base_sample)
            base_sample = tf.transpose(base_sample, perm=[1, 0, 2])
            base_log_prob = tf.transpose(base_log_prob, perm=[1, 0])
        else:
            base_sample = base.sample([n_flow, sample_size])
            base_log_prob = base.log_prob(base_sample)

        # Get mlp outupt that indexes into the normalizing flow layers.
        # First reshape and then transpose is to keep correctcly index
        # from each examples output into its corresponding normalizing flow.
        gov_param = tf.reshape(
                self.nf_param_mlp.operator(input_tensor),
                [n_flow, self.num_nf_layer, self.out_dim * 2 + 1])
        gov_param = tf.transpose(gov_param, perm=[1, 0, 2])
        flow = MultiLayerPlanarFlow(
                dim=self.out_dim, num_layer=self.num_nf_layer, n_flow=n_flow,
                non_linearity=self.nf_non_linearity, gov_param=gov_param)
        sample = flow.operator(base_sample)
        log_prob = base_log_prob - flow.log_det_jacobian(base_sample)
        return sample, log_prob
