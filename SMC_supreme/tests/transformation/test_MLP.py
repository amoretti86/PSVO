# Copyright 2018 Moretteam, Columbia University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ==============================================================================
import unittest
import tensorflow as tf

from SMC_supreme.transformation.MLP import MLP_transformation

# NUM_TESTS : 3
RUN_FROM = 0
RUN_TO = 3
TESTS_TO_RUN = list(range(RUN_FROM, RUN_TO))


class TestMLP(tf.test.TestCase):
    """
    """

    def setUp(self):
        """
        """
        print()
        tf.reset_default_graph()

    @unittest.skipIf(0 not in TESTS_TO_RUN, "Skipping")
    def test_MLP_init(self):
        """
        """
        print("Test 0: MLP initialization")

        MLP = MLP_transformation([5, 5], 3)

        self.assertEqual(MLP.mu_layer.compute_output_shape([1, 10]),
                         (1, 3),
                         "MLP declaration error")

    @unittest.skipIf(1 not in TESTS_TO_RUN, "Skipping")
    def test_MLP_transform(self):
        """
        """
        print("Test 1: MLP transform")

        MLP = MLP_transformation([5, 5], 3)

        X = tf.placeholder(tf.float64, [1, 10], 'X')
        mu, _ = MLP.transform(X)

        self.assertEqual(mu.shape, (1, 3),
                         "MLP transform dimension error")

    @unittest.skipIf(2 not in TESTS_TO_RUN, "Skipping")
    def test_MLP_cov(self):
        """
        """
        print("Test 2: MLP output covariance")

        MLP = MLP_transformation([5, 5], 3,
                                 output_cov=True)

        X = tf.placeholder(tf.float64, [1, 10], 'X')
        _, cov = MLP.transform(X)

        self.assertEqual(cov.shape, (1, 3, 3),
                         "MLP transform dimension error")

    def test_MLP_get_variables(self):
        """
        """
        print("Test 3: get variables")
        idim, h0dim, h1dim = 10, 5, 5

        MLP = MLP_transformation([h0dim, h1dim], 3,
                                 output_cov=True)
        X = tf.placeholder(tf.float64, [1, idim], 'X')
        MLP.transform(X)

        variables = MLP.get_variables()

        h0 = MLP.hidden_layers[0]
        self.assertEqual(variables[h0.name + '/weights'].shape,
                         (idim, h0dim),
                         "NN weights shape error")


if __name__ == "__main__":
    unittest.main(failfast=True)
