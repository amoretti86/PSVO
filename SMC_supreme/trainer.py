import numpy as np
from sklearn.utils import shuffle
import math

import tensorflow as tf
import os
import pickle
import time
import pdb

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from rslts_saving.rslts_saving import plot_R_square_epoch

SNR_SAMPLE_NUM = 100


class trainer:
    def __init__(self,
                 Dx, Dy,
                 n_particles, time,
                 batch_size, lr, epoch,
                 MSE_steps,
                 smoothing_perc_factor):
        self.Dx = Dx
        self.Dy = Dy

        self.n_particles = n_particles
        self.time = time

        self.batch_size = batch_size
        self.lr = lr
        self.epoch = epoch

        self.MSE_steps = MSE_steps

        self.save_res = False
        self.draw_quiver_during_training = False

        self.smoothing_perc_factor = smoothing_perc_factor

        # Used for SNR calculations
        # Only used for fhn data, and save_gradients = true
        self.have_surpassed_minus_365 = False
        self.have_surpassed_minus_220 = False
        # self.have_surpassed_minus_750 = False
        self.save_count_down = -1

    def training_params(self, early_stop_patience, lr_reduce_factor, lr_reduce_patience, min_lr, dropout_rate):
        self.early_stop_patience = early_stop_patience
        self.bestCost = 0
        self.early_stop_count = 0

        self.lr_reduce_factor = lr_reduce_factor
        self.lr_reduce_patience = lr_reduce_patience
        self.min_lr = min_lr
        self.lr_reduce_count = 0

        # dropout only used when training attention mechanism
        self.dropout_rate = dropout_rate

    def set_epoch_data_saving(self, RLT_DIR, saving_num, save_trajectory, save_y_hat, save_gradient):
        self.save_res = True
        self.RLT_DIR = RLT_DIR
        self.saving_num = saving_num
        self.save_trajectory = save_trajectory
        self.save_y_hat = save_y_hat
        self.save_gradient = save_gradient

        epoch_data_DIR = self.RLT_DIR.split("/")
        epoch_data_DIR.insert(epoch_data_DIR.index("rslts") + 1, "epoch_data")
        self.epoch_data_DIR = "/".join(epoch_data_DIR)

    def set_model_saving(self, save_tensorboard=False, save_model=False):
        self.save_tensorboard = save_tensorboard
        self.save_model = save_model
        if save_tensorboard:
            self.writer = tf.summary.FileWriter(self.RLT_DIR)

    def set_SMC(self, SMC):
        self.SMC = SMC

    def set_placeholders(self, obs, hidden, Input, dropout, smoothing_perc):
        self.obs = obs
        self.hidden = hidden
        self.Input = Input
        self.dropout = dropout
        self.smoothing_perc = smoothing_perc

    def set_quiver_arg(self, nextX, lattice, lattice_shape):
        self.nextX = nextX
        self.lattice = lattice
        self.lattice_shape = lattice_shape
        self.draw_quiver_during_training = True

    def evaluate(self, fetches, feed_dict_w_batches={}, average=False):
        """
        fetches: a single tensor or list of tensor to evaluate
        feed_dict_w_batches: {placeholder:input of multiple batches}
        """
        if not feed_dict_w_batches:
            return self.sess.run(fetches)

        n_batches = len(list(feed_dict_w_batches.values())[0])
        assert n_batches >= self.batch_size

        fetches_list = []
        feed_dict = {}
        for i in range(0, n_batches, self.batch_size):
            for key, value in feed_dict_w_batches.items():
                feed_dict[key] = value[i:i + self.batch_size]
            fetches_val = self.sess.run(fetches, feed_dict=feed_dict)
            fetches_list.append(fetches_val)

        res = []
        if isinstance(fetches, list):
            for i in range(len(fetches)):
                if isinstance(fetches_list[0][i], np.ndarray):
                    tmp = np.stack([x[i] for x in fetches_list])
                else:
                    tmp = np.array([x[i] for x in fetches_list])
                res.append(tmp)
        else:
            if isinstance(fetches_list[0], np.ndarray):
                res = np.stack(fetches_list)
            else:
                res = np.array(fetches_list)

        if average:
            if isinstance(res, list):
                res = [np.mean(x, axis=0) for x in res]
            else:
                res = np.mean(res, axis=0)

        return res

    def evaluate_R_square(self, MSE_ks, y_means, y_vars, hidden_set, obs_set, input_set):
        n_steps = y_means.shape.as_list()[0] - 1
        Dy = y_means.shape.as_list()[1]
        batch_size = self.batch_size
        n_batches = hidden_set.shape[0]

        combined_MSE_ks = np.zeros((n_steps + 1))             # combined MSE_ks across all batches
        combined_y_means = np.zeros((n_steps + 1, Dy))        # combined y_means across all batches
        combined_y_vars = np.zeros((n_steps + 1, Dy))         # combined y_vars across all batches

        for i in range(0, n_batches, batch_size):

            batch_MSE_ks, batch_y_means, batch_y_vars = self.sess.run([MSE_ks, y_means, y_vars],
                                                                      {self.obs: obs_set[i:i + batch_size],
                                                                       self.hidden: hidden_set[i:i + batch_size],
                                                                       self.Input: input_set[i:i + batch_size],
                                                                       self.dropout: np.zeros(batch_size),
                                                                       self.smoothing_perc: np.ones(batch_size)})
            # batch_MSE_ks.shape = (n_steps + 1)
            # batch_y_means.shape = (n_steps + 1, Dy)
            # batch_y_vars.shape = (n_steps + 1, Dy)

            # update combined_MSE_ks just by summing them across all batches
            combined_MSE_ks += batch_MSE_ks

            # update combined y_means and combined y_vars according to:
            # https://stats.stackexchange.com/questions/55999/is-it-possible-to-find-the-combined-standard-deviation
            Tmks = np.arange(self.time - n_steps, self.time + 1, 1)  # [time - n_steps, time - n_steps + 1, ..., time]
            Tmks = Tmks[-1:None:-1]                                  # [time, ..., time - n_steps + 1, time - n_steps]
            TmkxDy = np.tile(Tmks, (Dy, 1)).T                        # (n_steps + 1, Dy)

            # for k = 0, ..., n_steps,
            # its n1 = (time - k) * i, n2 = (time - k) * batch_size respectively
            n1 = TmkxDy * i                                     # (n_steps + 1, Dy)
            n2 = TmkxDy * batch_size                            # (n_steps + 1, Dy)

            combined_y_means_new = (n1 * combined_y_means + n2 * batch_y_means) / (n1 + n2)
            combined_y_vars = combined_y_vars + batch_y_vars + \
                n1 * (combined_y_means - combined_y_means_new)**2 + \
                n2 * (batch_y_means - combined_y_means_new)**2

            combined_y_means = combined_y_means_new

        combined_y_vars = np.mean(combined_y_vars, axis=1)
        R_square = 1 - combined_MSE_ks / combined_y_vars
        mean_MSE_ks = combined_MSE_ks / (Tmks * n_batches)

        return mean_MSE_ks, R_square

    def set_up_gradient(self, loss):
        trans_variables_dict = {}
        SMC = self.SMC
        q0 = SMC.q0.transformation
        q1 = SMC.q1.transformation
        g = SMC.g.transformation
        q2 = None if self.SMC.q2 is None else SMC.q2.transformation  # If not using 2q network, q2 == None
        f = None if self.SMC.f == self.SMC.q1 else SMC.f.transformation  # If using bootstrap, f == q1

        for MLP_trans in [q0, q1, q2, f, g]:
            if MLP_trans is None:
                continue

            variables_dict = MLP_trans.get_variables()
            for key, val in variables_dict.items():
                trans_variables_dict[MLP_trans.name + "/" + key] = val

        variable_names = list(trans_variables_dict.keys())
        variables = list(trans_variables_dict.values())
        gradients = tf.gradients(loss, variables)

        self.gradients_dict = dict(zip(variable_names, gradients))

    def evaluate_gradients(self, feed_dict):
        variable_names = list(self.gradients_dict.keys())
        gradients = list(self.gradients_dict.values())
        gradients_val_samples = [self.evaluate(gradients, feed_dict, average=True) for _ in range(SNR_SAMPLE_NUM)]
        gradients_val = [np.stack([gradients_val_sample[i] for gradients_val_sample in gradients_val_samples])
                         for i in range(len(gradients))]
        res_dict = dict(zip(variable_names, gradients_val))

        return res_dict

    def compute_gradient(self, loss, feed_dict):
        # set up gradient nodes
        gradients_dict = {}
        SMC = self.SMC
        q0 = SMC.q0.transformation
        q1 = SMC.q1.transformation
        g = SMC.g.transformation
        q2 = None if self.SMC.q2 is None else SMC.q2.transformation         # If not using 2q network, q2 == None
        f = None if self.SMC.f == self.SMC.q1 else SMC.f.transformation     # If using bootstrap, f == q1

        for MLP_trans in [q0, q1, q2, f, g]:
            if MLP_trans is None:
                continue

            variables_dict = MLP_trans.get_variables()
            variable_names = list(variables_dict.keys())
            variables = list(variables_dict.values())
            gradients = [tf.gradients(loss, variable) for variable in variables]

            gradients_dict[MLP_trans.name] = dict(zip(variable_names, gradients))

        # evaluate gradient value
        list_of_gradients_val_dict = []
        for i in SNR_SAMPLE_NUM:
            gradients_val_dict = {}
            for MLP_name, MLP_gradients_dict in gradients_dict.items():
                variable_names = list(MLP_gradients_dict.keys())
                gradients = list(MLP_gradients_dict.values())
                gradients_val = [self.evaluate(gradient, feed_dict, average=True) for gradient in gradients]

                gradients_val_dict[MLP_name] = dict(zip(variable_names, gradients_val))

            list_of_gradients_val_dict.append(gradients_val_dict)

        return list_of_gradients_val_dict

    def train(self,
              obs_train, obs_test,
              hidden_train, hidden_test,
              input_train, input_test,
              print_freq):

        log_ZSMC, log = self.SMC.get_log_ZSMC(self.obs, self.hidden, self.Input)

        # n_step_MSE now takes Xs as input rather than self.hidden
        # so there is no need to evalute enumerical value of Xs and feed it into self.hidden
        Xs = log["Xs"]
        MSE_ks, y_means, y_vars, y_hat = self.SMC.n_step_MSE(self.MSE_steps, Xs, self.obs, self.Input)

        # used to compare signal-to-noise ratio of gradient for different number of paricles
        self.set_up_gradient(log_ZSMC)

        with tf.variable_scope("train"):
            lr = tf.placeholder(tf.float32, name="lr")
            train_op = tf.train.AdamOptimizer(lr).minimize(-log_ZSMC)

        init = tf.global_variables_initializer()

        if self.save_res:
            saver = tf.train.Saver(max_to_keep=1)

            log_ZSMC_trains = []
            log_ZSMC_tests = []
            MSE_trains = []
            MSE_tests = []
            R_square_trains = []
            R_square_tests = []

        self.sess = tf.Session()

        print("initializing variables...")
        self.sess.run(init)

        # unused tensorboard stuff
        if self.save_res and self.save_tensorboard:
            self.writer.add_graph(self.sess.graph)

        log_ZSMC_train = self.evaluate(log_ZSMC,
                                       {self.obs:            obs_train,
                                        self.hidden:         hidden_train,
                                        self.Input:          input_train,
                                        self.dropout:        np.zeros(len(obs_train)),
                                        self.smoothing_perc: np.zeros(len(obs_train))},
                                       average=True)
        log_ZSMC_test = self.evaluate(log_ZSMC,
                                      {self.obs:            obs_test,
                                       self.hidden:         hidden_test,
                                       self.Input:          input_test,
                                       self.dropout:        np.zeros(len(obs_test)),
                                       self.smoothing_perc: np.zeros(len(obs_test))},
                                      average=True)

        MSE_train, R_square_train = self.evaluate_R_square(MSE_ks, y_means, y_vars,
                                                           hidden_train, obs_train, input_train)
        MSE_test, R_square_test = self.evaluate_R_square(MSE_ks, y_means, y_vars,
                                                         hidden_test, obs_test, input_test)

        print("iter {:>3}, train log_ZSMC: {:>7.3f}, test log_ZSMC: {:>7.3f}"
              .format(0, log_ZSMC_train, log_ZSMC_test))

        if self.save_res:
            log_ZSMC_trains.append(log_ZSMC_train)
            log_ZSMC_tests.append(log_ZSMC_test)
            MSE_trains.append(MSE_train)
            MSE_tests.append(MSE_test)
            R_square_trains.append(R_square_train)
            R_square_tests.append(R_square_test)

        for i in range(self.epoch):
            start = time.time()

            if i < self.epoch * 3 / 4:
                smoothing_perc_epoch = 1 - (1 - i / self.epoch) ** self.smoothing_perc_factor
            else:
                smoothing_perc_epoch = 1
            # self.lr = (self.start_lr - i/self.epoch*(self.lr - self.end_lr))

            obs_train, hidden_train = shuffle(obs_train, hidden_train)
            for j in range(0, len(obs_train), self.batch_size):
                self.sess.run(train_op,
                              feed_dict={self.obs:            obs_train[j:j + self.batch_size],
                                         self.hidden:         hidden_train[j:j + self.batch_size],
                                         self.Input:          input_train[j:j + self.batch_size],
                                         self.dropout:        np.ones(self.batch_size) * self.dropout_rate,
                                         self.smoothing_perc: np.ones(self.batch_size) * smoothing_perc_epoch,
                                         lr:                  self.lr})

            # print training and testing loss
            if (i + 1) % print_freq == 0:
                log_ZSMC_train = self.evaluate(log_ZSMC,
                                               {self.obs:            obs_train,
                                                self.hidden:         hidden_train,
                                                self.Input:          input_train,
                                                self.dropout:        np.zeros(len(obs_train)),
                                                self.smoothing_perc: np.ones(len(obs_train)) * smoothing_perc_epoch},
                                               average=True)
                log_ZSMC_test = self.evaluate(log_ZSMC,
                                              {self.obs:            obs_test,
                                               self.hidden:         hidden_test,
                                               self.Input:          input_test,
                                               self.dropout:        np.zeros(len(obs_test)),
                                               self.smoothing_perc: np.ones(len(obs_test)) * smoothing_perc_epoch},
                                              average=True)

                MSE_train, R_square_train = self.evaluate_R_square(MSE_ks, y_means, y_vars,
                                                                   hidden_train, obs_train, input_train)
                MSE_test, R_square_test = self.evaluate_R_square(MSE_ks, y_means, y_vars,
                                                                 hidden_test, obs_test, input_test)

                print()
                print("iter", i + 1)
                print("Train log_ZSMC: {:>7.3f}, valid log_ZSMC: {:>7.3f}"
                      .format(log_ZSMC_train, log_ZSMC_test))

                print("Train, Valid k-step Rsq:\n", R_square_train, "\n", R_square_test)

                if not math.isfinite(log_ZSMC_train):
                    break

                # save useful data
                if self.save_res:
                    log_ZSMC_trains.append(log_ZSMC_train)
                    log_ZSMC_tests.append(log_ZSMC_test)
                    MSE_trains.append(MSE_train)
                    MSE_tests.append(MSE_test)
                    R_square_trains.append(R_square_train)
                    R_square_tests.append(R_square_test)

                    plot_R_square_epoch(self.RLT_DIR, R_square_trains[-1], R_square_tests[-1], i + 1)

                    if not os.path.exists(self.epoch_data_DIR):
                        os.makedirs(self.epoch_data_DIR)
                    metric_dict = {"log_ZSMC_train": log_ZSMC_train,
                                   "log_ZSMC_test":  log_ZSMC_test,
                                   "R_square_train": R_square_train,
                                   "R_square_test":  R_square_test}
                    with open(self.epoch_data_DIR + "metric_{}.p".format(i + 1), "wb") as f:
                        pickle.dump(metric_dict, f)
                    if self.save_gradient:

                        # an ad-hoc and naive way to determine whether the fhn experiment is near convergence
                        # please tell me if you have any other ideas

                        # if loss gets > -365, save gradients for 20 epochs

                        # will delete later, used for test
                        # if not self.have_surpassed_minus_750 and log_ZSMC_test > -1750:
                        #    print("Surpassing -1750, collecting the gradients....\n")
                        #    self.have_surpassed_minus_750 = True
                        #    self.save_count_down = 20

                        if not self.have_surpassed_minus_365 and log_ZSMC_test > -365:
                            self.have_surpassed_minus_365 = True
                            self.save_count_down = 20

                        # if loss gets > -220, save gradients for 20 epochs
                        if not self.have_surpassed_minus_220 and log_ZSMC_test > -220:
                            self.have_surpassed_minus_220 = True
                            self.save_count_down = 20

                        if self.save_count_down > 0:
                            self.save_count_down -= 1
                            print("Count down: ", self.save_count_down)
                            gradients_feed_dict = {self.obs:            obs_train[0:self.saving_num],
                                                   self.hidden:         hidden_train[0:self.saving_num],
                                                   self.Input:          input_train[0:self.saving_num],
                                                   self.dropout:        np.zeros(self.saving_num),
                                                   self.smoothing_perc: np.ones(self.saving_num) * smoothing_perc_epoch}

                            gradients_val_dict = self.evaluate_gradients(gradients_feed_dict)

                            with open(self.epoch_data_DIR + "gradient_{}.p".format(i + 1), "wb") as f:
                                pickle.dump(gradients_val_dict, f)

                    # ------------ end of saving gradient ------------------

                    evaluate_feed_dict = {self.obs:             obs_test[0:self.saving_num],
                                          self.hidden:          hidden_test[0:self.saving_num],
                                          self.Input:           input_test[0:self.saving_num],
                                          self.dropout:         np.zeros(self.saving_num),
                                          self.smoothing_perc:  np.ones(self.saving_num)}
                    if self.save_trajectory:
                        Xs_val = self.evaluate(Xs, evaluate_feed_dict, average=False)
                        Xs_val = Xs_val.reshape([-1] + list(Xs_val.shape[2:]))
                        trajectory_dict = {"Xs": Xs_val}
                        with open(self.epoch_data_DIR + "trajectory_{}.p".format(i + 1), "wb") as f:
                            pickle.dump(trajectory_dict, f)

                    if self.save_y_hat:
                        y_hat_val = self.evaluate(y_hat, evaluate_feed_dict, average=False)
                        y_hat_val = [step_y_hat_val.reshape([-1] + list(step_y_hat_val.shape[2:]))
                                     for step_y_hat_val in y_hat_val]
                        y_hat_dict = {"y_hat": y_hat_val}
                        with open(self.epoch_data_DIR + "y_hat_{}.p".format(i + 1), "wb") as f:
                            pickle.dump(y_hat_dict, f)

                if self.draw_quiver_during_training:
                    if self.Dx == 2:
                        self.draw_2D_quiver_plot(Xs_val, self.nextX, self.lattice, i + 1)
                    elif self.Dx == 3:
                        self.draw_3D_quiver_plot(Xs_val, i + 1)

                # determine whether should decrease lr or even stop training
                cost = np.array(log_ZSMC_tests)
                if self.bestCost != np.argmax(cost):
                    self.early_stop_count = 0
                    self.lr_reduce_count = 0
                    self.bestCost = np.argmax(cost)

                print("best valid cost on iter: {}\n".format(self.bestCost * print_freq))

                if self.bestCost < np.int((i + 1) / print_freq):
                    self.early_stop_count += 1
                    if self.early_stop_count > self.early_stop_patience:
                        print("valid cost not improving. stopping training...")
                        break

                    self.lr_reduce_count += 1
                    if self.lr_reduce_count * print_freq >= self.lr_reduce_patience:
                        self.lr_reduce_count = 0
                        self.lr = max(self.lr * self.lr_reduce_factor, self.min_lr)
                        print("valid cost not improving. reduce learning rate to {}".format(self.lr))

                if self.save_model:
                    if not os.path.exists(self.RLT_DIR + "model/"):
                        os.makedirs(self.RLT_DIR + "model/")
                    saver.save(self.sess, self.RLT_DIR + "model/model_epoch", global_step=i + 1)

            end = time.time()
            print("epoch {:<4} took {:.3f} seconds".format(i + 1, end - start))

        print("finished training...")

        metrics = {"log_ZSMC_trains": log_ZSMC_trains,
                   "log_ZSMC_tests": log_ZSMC_tests,
                   "MSE_trains": MSE_trains,
                   "MSE_tests": MSE_tests,
                   "R_square_trains": R_square_trains,
                   "R_square_tests": R_square_tests}
        log["y_hat"] = y_hat

        return metrics, log

    def close_session(self):
        self.sess.close()

    def draw_2D_quiver_plot(self, Xs_val, nextX, lattice, epoch):
        # Xs_val.shape = (saving_num, time, n_particles, Dx)
        X_trajs = np.mean(Xs_val, axis=2)

        plt.figure()
        for X_traj in X_trajs[0:self.saving_num]:
            plt.plot(X_traj[:, 0], X_traj[:, 1])
            plt.scatter(X_traj[0, 0], X_traj[0, 1])
        plt.title("quiver")
        plt.xlabel("x_dim 1")
        plt.ylabel("x_dim 2")

        if nextX is not None:
            axes = plt.gca()
            x1range, x2range = axes.get_xlim(), axes.get_ylim()
            X = lattice_val = self.define2Dlattice(x1range, x2range)

            nextX = self.sess.run(nextX, feed_dict={lattice: lattice_val})

            scale = int(5 / 3 * max(abs(x1range[0]) + abs(x1range[1]), abs(x2range[0]) + abs(x2range[1])))
            plt.quiver(X[:, :, 0], X[:, :, 1], nextX[:, :, 0] - X[:, :, 0], nextX[:, :, 1] - X[:, :, 1], scale=scale)

            quiver_dict = {"X_trajs": X_trajs, "X": X, "nextX": nextX}
            with open(self.epoch_data_DIR + "lattice_val_{}.p".format(epoch), "wb") as f:
                pickle.dump(quiver_dict, f)

        # sns.despine()
        if not os.path.exists(self.RLT_DIR + "quiver/"):
            os.makedirs(self.RLT_DIR + "quiver/")
        plt.savefig(self.RLT_DIR + "quiver/epoch_{}".format(epoch))
        plt.close()

    def define2Dlattice(self, x1range=(-30.0, 30.0), x2range=(-30.0, 30.)):

        x1coords = np.linspace(x1range[0], x1range[1], num=self.lattice_shape[0])
        x2coords = np.linspace(x2range[0], x2range[1], num=self.lattice_shape[1])
        Xlattice = np.stack(np.meshgrid(x1coords, x2coords), axis=-1)
        return Xlattice

    def draw_3D_quiver_plot(self, Xs_val, epoch):
        # Xs_val.shape = (saving_num, time, n_particles, Dx)
        X_trajs = np.mean(Xs_val, axis=2)

        fig = plt.figure()
        ax = fig.gca(projection="3d")
        plt.title("hidden state for all particles")
        ax.set_xlabel("x_dim 1")
        ax.set_ylabel("x_dim 2")
        ax.set_zlabel("x_dim 3")
        for X_traj in X_trajs:
            ax.plot(X_traj[:, 0], X_traj[:, 1], X_traj[:, 2])
            ax.scatter(X_traj[0, 0], X_traj[0, 1], X_traj[0, 2])

        if not os.path.exists(self.RLT_DIR + "quiver/"):
            os.makedirs(self.RLT_DIR + "quiver/")
        for angle in range(45, 360, 45):
            ax.view_init(30, angle)
            plt.savefig(self.RLT_DIR + "quiver/epoch_{}_angle_{}".format(epoch, angle))
        plt.close()
