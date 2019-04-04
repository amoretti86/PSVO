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

from SMC_supreme.model import SSM
from SMC_supreme.runner_flag import FLAGS

# NUM_TESTS : 1
RUN_FROM = 0
RUN_TO = 1
TESTS_TO_RUN = list(range(RUN_FROM, RUN_TO))


class TestSSMInitialization(tf.test.TestCase):
    """
    """

    def setUp(self):
        """
        """
        print()
        tf.reset_default_graph()

    @unittest.skipIf(0 not in TESTS_TO_RUN, "Skipping")
    def test_SSM_init(self):
        """
        """
        print("Test 0: SSM initialization")
        Dx = FLAGS.Dx

        SSM_model = SSM(FLAGS)

        self.assertEqual(Dx, SSM_model.Dx, "Failed flag assignment")


if __name__ == "__main__":
    unittest.main(failfast=True)
