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


class trainer:
    def __init__(self,
                 Dx, Dy,
                 n_particles, time,
                 batch_size, lr, epoch,
                 MSE_steps,
                 beta,
                 maxNumberNoImprovement,
                 x_0_learnable,
                 smoothing_perc_factor):
        self.Dx = Dx
        self.Dy = Dy

        self.n_particles = n_particles
        self.time = time

        self.batch_size = batch_size
        self.lr = lr
        self.epoch = epoch

        self.MSE_steps = MSE_steps

        self.store_res = False
        self.draw_quiver_during_training = False

        self.beta = beta
        self.maxNumberNoImprovement = maxNumberNoImprovement
        self.bestCost = -np.inf
        self.costUpdate = 0

        self.x_0_learnable = x_0_learnable

        self.smoothing_perc_factor = smoothing_perc_factor

    def set_rslt_saving(self, RLT_DIR, save_freq, saving_num, save_tensorboard=False, save_model=False):
        self.store_res = True
        self.RLT_DIR = RLT_DIR
        self.save_freq = save_freq
        self.saving_num = saving_num
        self.save_tensorboard = save_tensorboard
        self.save_model = save_model
        if save_tensorboard:
            self.writer = tf.summary.FileWriter(RLT_DIR)

        epoch_data_DIR = self.RLT_DIR.split("/")
        epoch_data_DIR.insert(epoch_data_DIR.index("rslts") + 1, "epoch_data")
        self.epoch_data_DIR = "/".join(epoch_data_DIR)

    def set_SMC(self, SMC_train):
        self.SMC_train = SMC_train

    def set_placeholders(self, x_0, obs, hidden, smoothing_perc):
        self.x_0 = x_0
        self.obs = obs
        self.hidden = hidden
        self.smoothing_perc = smoothing_perc

    def set_quiver_arg(self, nextX, lattice, quiver_traj_num, lattice_shape):
        self.nextX = nextX
        self.lattice = lattice
        self.quiver_traj_num = quiver_traj_num
        self.lattice_shape = lattice_shape
        self.draw_quiver_during_training = True

    def evaluate(self, fetches, feed_dict_w_batches={}, average=False):
        """
        fetches: a single tensor or list of tensor to evaluate
        feed_dict: {placeholder:input of multiple batches}
        """
        if not feed_dict_w_batches:
            return self.sess.run(fetches)

        n_batches = len(list(feed_dict_w_batches.values())[0])
        assert n_batches % self.batch_size == 0

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
                    tmp = np.concatenate([x[i] for x in fetches_list])
                else:
                    tmp = np.array([x[i] for x in fetches_list])
                res.append(tmp)
        else:
            if isinstance(fetches_list[0], np.ndarray):
                res = np.concatenate(fetches_list)
            else:
                res = np.array(fetches_list)

        if average:
            if isinstance(res, list):
                res = [np.mean(x, axis=0) for x in res]
            else:
                res = np.mean(res, axis=0)

        return res

    def evaluate_R_square(self, MSE_ks, y_means, y_vars, hidden_set, obs_set, is_test=False):
        n_steps = y_means.shape.as_list()[0] - 1
        Dy = y_means.shape.as_list()[1]
        batch_size = self.batch_size
        n_batches = hidden_set.shape[0]

        combined_MSE_ks = np.zeros((n_steps + 1))             # combined MSE_ks across all batches
        combined_y_means = np.zeros((n_steps + 1, Dy))        # combined y_means across all batches
        combined_y_vars = np.zeros((n_steps + 1, Dy))         # combined y_vars across all batches

        for i in range(0, n_batches, batch_size):

            if self.x_0_learnable:
                offset = self.n_train if is_test else 0
                x_0_feed = np.arange(i, i + batch_size) + offset
            else:
                x_0_feed = hidden_set[i:i + batch_size, 0]

            batch_MSE_ks, batch_y_means, batch_y_vars = self.sess.run([MSE_ks, y_means, y_vars],
                                                                      {self.obs: obs_set[i:i + batch_size],
                                                                       self.x_0: x_0_feed,
                                                                       self.hidden: hidden_set[i:i + batch_size],
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

    def train(self, obs_train, obs_test, print_freq, hidden_train, hidden_test):
        self.n_train = n_train = len(obs_train)
        self.n_test = n_test = len(obs_test)

        log_ZSMC, log = self.SMC_train.get_log_ZSMC(self.obs, self.hidden)

        # n_step_MSE now takes Xs as input rather than self.hidden
        # so there is no need to evalute enumerical value of Xs and feed it into self.hidden
        Xs = log[0]
        MSE_ks, y_means, y_vars, y_hat = self.SMC_train.n_step_MSE(self.MSE_steps, Xs, self.obs)

        cost_w_MSE = -log_ZSMC  # + self.beta * MSE_ks[0]
        with tf.variable_scope("train"):
            train_op = tf.train.AdamOptimizer(self.lr).minimize(cost_w_MSE)

        init = tf.global_variables_initializer()

        if self.store_res:
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

        if self.store_res and self.save_tensorboard:
            self.writer.add_graph(self.sess.graph)

        # print("trainable_variables:")
        # for tnsr in tf.trainable_variables():
        #     print("\t", tnsr)

        if self.x_0_learnable:
            x_0_feed_train = np.arange(n_train)
            x_0_feed_test = n_train + np.arange(n_test)
        else:
            x_0_feed_train = hidden_train[:, 0]
            x_0_feed_test = hidden_test[:, 0]

        log_ZSMC_train = self.evaluate(log_ZSMC,
                                       {self.obs: obs_train,
                                        self.x_0: x_0_feed_train,
                                        self.hidden: hidden_train,
                                        self.smoothing_perc: np.zeros(len(obs_train))},
                                       average=True)
        log_ZSMC_test = self.evaluate(log_ZSMC,
                                      {self.obs: obs_test,
                                       self.x_0: x_0_feed_test,
                                       self.hidden: hidden_test,
                                       self.smoothing_perc: np.zeros(len(obs_test))},
                                      average=True)

        MSE_train, R_square_train = self.evaluate_R_square(MSE_ks, y_means, y_vars,
                                                           hidden_train, obs_train, is_test=False)
        MSE_test, R_square_test = self.evaluate_R_square(MSE_ks, y_means, y_vars,
                                                         hidden_test, obs_test, is_test=True)

        print("iter {:>3}, train log_ZSMC: {:>7.3f}, test log_ZSMC: {:>7.3f}"
              .format(0, log_ZSMC_train, log_ZSMC_test))

        if self.store_res:
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
                if self.x_0_learnable:
                    x_0_feed = np.arange(j, j + self.batch_size)
                else:
                    x_0_feed = hidden_train[j:j + self.batch_size, 0]
                self.sess.run(train_op,
                              feed_dict={self.obs: obs_train[j:j + self.batch_size],
                                         self.x_0: x_0_feed,
                                         self.hidden: hidden_train[j:j + self.batch_size],
                                         self.smoothing_perc: np.ones(self.batch_size) * smoothing_perc_epoch})

            # print training and testing loss
            if (i + 1) % print_freq == 0:
                log_ZSMC_train = self.evaluate(log_ZSMC,
                                               {self.obs: obs_train,
                                                self.x_0: x_0_feed_train,
                                                self.hidden: hidden_train,
                                                self.smoothing_perc: np.ones(len(obs_train)) * smoothing_perc_epoch},
                                               average=True)
                log_ZSMC_test = self.evaluate(log_ZSMC,
                                              {self.obs: obs_test,
                                               self.x_0: x_0_feed_test,
                                               self.hidden: hidden_test,
                                               self.smoothing_perc: np.ones(len(obs_test)) * smoothing_perc_epoch},
                                              average=True)

                MSE_train, R_square_train = self.evaluate_R_square(MSE_ks, y_means, y_vars,
                                                                   hidden_train, obs_train, is_test=False)
                MSE_test, R_square_test = self.evaluate_R_square(MSE_ks, y_means, y_vars,
                                                                 hidden_test, obs_test, is_test=True)

                print("iter {:>3}, train log_ZSMC: {:>7.3f}, valid log_ZSMC: {:>7.3f}"
                      .format(i + 1, log_ZSMC_train, log_ZSMC_test))

                print("Train, Valid k-step Rsq:\n", R_square_train, "\n", R_square_test)

                if not math.isfinite(log_ZSMC_train):
                    break

                if self.store_res:
                    log_ZSMC_trains.append(log_ZSMC_train)
                    log_ZSMC_tests.append(log_ZSMC_test)
                    MSE_trains.append(MSE_train)
                    MSE_tests.append(MSE_test)
                    R_square_trains.append(R_square_train)
                    R_square_tests.append(R_square_test)

                    plot_R_square_epoch(self.RLT_DIR, R_square_trains[-1], R_square_tests[-1], i + 1)

                    Xs_val = self.evaluate(Xs,
                                           {self.obs: obs_test[0:self.saving_num],
                                            self.x_0: x_0_feed_test[0:self.saving_num],
                                            self.hidden: hidden_test[0:self.saving_num],
                                            self.smoothing_perc: np.ones(self.saving_num)},
                                           average=False)
                    y_hat_val = self.evaluate(y_hat,
                                              {self.obs: obs_test[0:self.saving_num],
                                               self.x_0: x_0_feed_test[0:self.saving_num],
                                               self.hidden: hidden_test[0:self.saving_num],
                                               self.smoothing_perc: np.ones(self.saving_num)},
                                              average=False)

                    epoch_dict = {"R_square_train": R_square_train[-1],
                                  "R_square_test": R_square_test[-1],
                                  "Xs_val": Xs_val,
                                  "y_hat_val": y_hat_val}

                    if not os.path.exists(self.epoch_data_DIR):
                        os.makedirs(self.epoch_data_DIR)
                    with open(self.epoch_data_DIR + "epoch_{}.p".format(i + 1), "wb") as f:
                        pickle.dump(epoch_dict, f)

                if self.draw_quiver_during_training:
                    if self.Dx == 2:
                        self.draw_2D_quiver_plot(Xs_val, self.nextX, self.lattice, i + 1)
                    elif self.Dx == 3:
                        self.draw_3D_quiver_plot(Xs_val, self.nextX, self.lattice, i + 1)

                # determine whether should stop training
                cost = np.array(log_ZSMC_tests) - self.beta * np.array([x[0] for x in MSE_tests])
                if self.bestCost != np.argmax(cost):
                    self.costUpdate = 0
                    self.bestCost = np.argmax(cost)

                print("best valid cost on iter:", self.bestCost * print_freq)

                if self.bestCost < np.int((i + 1) / print_freq):
                    self.costUpdate += 1
                    if self.costUpdate > self.maxNumberNoImprovement:
                        print("valid cost not improving. stopping training...")
                        break

            if self.store_res and self.save_model and (i + 1) % self.save_freq == 0:
                if not os.path.exists(self.RLT_DIR + "model/"):
                    os.makedirs(self.RLT_DIR + "model/")
                saver.save(self.sess, self.RLT_DIR + "model/model_epoch", global_step=i + 1)

            end = time.time()
            print("epoch {:<4} took {:.3f} seconds".format(i + 1, end - start))

        print("finished training...")

        losses = None
        if self.store_res:
            losses = [log_ZSMC_trains, log_ZSMC_tests,
                      MSE_trains, MSE_tests,
                      R_square_trains, R_square_tests]
        tensors = [log, y_hat]

        return losses, tensors

    def close_session(self):
        self.sess.close()

    def draw_2D_quiver_plot(self, Xs_val, nextX, lattice, epoch):
        # Xs_val.shape = (saving_num, time, n_particles, Dx)
        X_trajs = np.mean(Xs_val, axis=2)

        plt.figure()
        for X_traj in X_trajs[0:self.quiver_traj_num]:
            plt.plot(X_traj[:, 0], X_traj[:, 1])
            plt.scatter(X_traj[0, 0], X_traj[0, 1])
        plt.title("quiver")
        plt.xlabel("x_dim 1")
        plt.ylabel("x_dim 2")
        axes = plt.gca()
        x1range, x2range = axes.get_xlim(), axes.get_ylim()

        lattice_val = self.define2Dlattice(x1range, x2range)

        X = lattice_val
        nextX = self.sess.run(nextX, feed_dict={lattice: lattice_val})

        scale = int(5 / 3 * max(abs(x1range[0]) + abs(x1range[1]), abs(x2range[0]) + abs(x2range[1])))
        plt.quiver(X[:, :, 0], X[:, :, 1], nextX[:, :, 0] - X[:, :, 0], nextX[:, :, 1] - X[:, :, 1], scale=scale)

        # sns.despine()
        if not os.path.exists(self.RLT_DIR + "quiver/"):
            os.makedirs(self.RLT_DIR + "quiver/")
        plt.savefig(self.RLT_DIR + "quiver/epoch_{}".format(epoch))
        plt.close()

        quiver_dict = {"X_trajs": X_trajs, "X": X, "nextX": nextX}
        with open(self.epoch_data_DIR + "lattice_val_{}.p".format(epoch), "wb") as f:
            pickle.dump(quiver_dict, f)

    def define2Dlattice(self, x1range=(-30.0, 30.0), x2range=(-30.0, 30.)):

        x1coords = np.linspace(x1range[0], x1range[1], num=self.lattice_shape[0])
        x2coords = np.linspace(x2range[0], x2range[1], num=self.lattice_shape[1])
        Xlattice = np.stack(np.meshgrid(x1coords, x2coords), axis=-1)
        return Xlattice

    def draw_3D_quiver_plot(self, Xs_val, nextX, lattice, epoch):
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
