import subprocess
import itertools
import os
import time

# This is design for python 2.6
if __name__ == "__main__":

    mem = 60                        # memory in Gb
    sh_time = 60                    # time in hour
    task_name = "fhn_SNR"            # name of the task
    conda_path = "/ifs/scratch/c2b2/ip_lab/lw2827/miniconda3/bin/"
    env_name = "aesmc_env"
    execution_path = "/ifs/scratch/c2b2/ip_lab/lw2827/"
    py_script_path = "/ifs/scratch/c2b2/ip_lab/lw2827/VISMC_SNR_branch/VISMC/SMC_SNR/runner_flag.py"

    sh_name = "run"

    params_dict = {}

    # --------------------- training hyperparameters --------------------- #
    params_dict["Dx"] = [2]
    params_dict["Dy"] = [1]
    params_dict["Di"] = [1]
    params_dict["n_particles"] = [1]


    params_dict["batch_size"] = [1]
    params_dict["lr"] = [2e-3]
    params_dict["epoch"] = [300]
    params_dict["seed"] = [0,20,30]

    # ----------------------- SNR experiment ----------------- #
    # save gradient should be set True
    params_dict["save_gradient"] = [True]

    params_dict["SNR_sample_num"] = [100]

    #SNR_NP_list = [1, 8, 64, 128, 512, 1024, 2048]
    params_dict["SNR_NP_list"] = [[1,2,4, 8,16, 32, 64, 128,256, 512, 1024,2048]]
    #SNR_NP_list = [1, 2]

    #SNR_collect_grads_points = [-1750, -650,-600, -550, -500, -450, -400, -350, -300, -250, -220]
    #SNR_collect_grads_points = [-700, -600,-500, -400, -350, -300, -250, -220]
    #params_dict["SNR_collect_grads_points"] = [[-700, -400, -350, -250, -220]]
    #params_dict["SNR_collect_grads_points"] = [[-700, -400]]
    params_dict["SNR_collect_grads_points"] = [[-350]]


    # ------------------ loss type ---------------------- #
    params_dict["loss_type"] = ['soft', 'soft2', 'main', 'full']

    # --------------------- data set parameters --------------------- #
    params_dict["generateTrainingData"] = [False]

    params_dict["datadir"] = ["/ifs/scratch/c2b2/ip_lab/lw2827/VISMC/data/fhn/[1,0]_obs_cov_0.01/"]
    # "/ifs/scratch/c2b2/ip_lab/zw2504/VISMC/data/fitzhughnagumo/"
    # "/ifs/scratch/c2b2/ip_lab/zw2504/VISMC/data/fhn/[1,0]_obs_cov_0.01/"
    # "/ifs/scratch/c2b2/ip_lab/zw2504/VISMC/data/fhn/changing/"
    params_dict["datadict"] = ["datadict"]
    params_dict["isPython2"] = [True]

    params_dict["time"] = [200]
    params_dict["n_train"] = [200]
    params_dict["n_test"] = [40]

    # --------------------- model parameters --------------------- #
    # Feed-Forward Network (FFN)
    params_dict["q0_layers"] = [[16]]
    params_dict["q1_layers"] = [[16]]
    params_dict["q2_layers"] = [[16]]
    params_dict["f_layers"] = [[16]]
    params_dict["g_layers"] = [[16]]

    params_dict["q0_sigma_init"] = [5]
    params_dict["q1_sigma_init"] = [5]
    params_dict["q2_sigma_init"] = [5]
    params_dict["f_sigma_init"] = [5]
    params_dict["g_sigma_init"] = [5]

    params_dict["q0_sigma_min"] = [1]
    params_dict["q1_sigma_min"] = [1]
    params_dict["q2_sigma_min"] = [1]
    params_dict["f_sigma_min"] = [1]
    params_dict["g_sigma_min"] = [1]

    # bidirectional RNN
    params_dict["y_smoother_Dhs"] = [[16]]
    params_dict["X0_smoother_Dhs"] = [[16]]

    # Self-Attention encoder
    params_dict["num_hidden_layers"] = [6]
    params_dict["num_heads"] = [16]
    params_dict["hidden_size"] = [64]
    params_dict["filter_size"] = [64]
    params_dict["dropout_rate"] = [0.1]

    # --------------------- FFN flags --------------------- #
    params_dict["use_bootstrap"] = [True]
    params_dict["q_uses_true_X"] = [False]
    params_dict["use_residual"] = [False]
    params_dict["use_2_q"] = [True]
    params_dict["output_cov"] = [False]
    params_dict["diag_cov"] = [True]
    params_dict["use_input"] = [False]

    # --------------------- FFBS flags --------------------- #
    params_dict["FFBS"] = [False]
    params_dict["smoothing_perc_factor"] = [0]
    params_dict["FFBS_to_learn"] = [False]

    # --------------------- smoother flags --------------------- #
    params_dict["smooth_obs"] = [False]
    params_dict["use_RNN"] = [True]
    params_dict["X0_use_separate_RNN"] = [True]
    params_dict["use_stack_rnn"] = [True]

    # --------------------- training flags --------------------- #
    params_dict["early_stop_patience"] = [200]
    params_dict["lr_reduce_patience"] = [200]
    params_dict["lr_reduce_factor"] = [0.7]
    params_dict["min_lr"] = [1e-5]
    params_dict["use_stop_gradient"] = [False]

    # --------------------- printing and data saving params --------------------- #
    params_dict["print_freq"] = [1]

    params_dict["store_res"] = [True]
    # params_dict["rslt_dir_name"] = "lorenz_1D"
    params_dict["MSE_steps"] = [30]

    # how many trajectories to draw in quiver plot
    params_dict["quiver_traj_num"] = [30]
    params_dict["lattice_shape"] = [[25, 25]]  # [25, 25] or [10, 10, 5]

    params_dict["saving_num"] = [30]

    params_dict["save_tensorboard"] = [False]
    params_dict["save_model"] = [False]
    params_dict["save_freq"] = [1]

    params_dict["save_gradient"] = [True]
    params_dict["save_y_hat"] = [False]

    # --------------------- parameters part ends --------------------- #
    param_keys = list(params_dict.keys())
    param_values = list(params_dict.values())
    param_vals_permutation = list(itertools.product(*param_values))

    for i, param_vals in enumerate(param_vals_permutation):
        args = ""
        arg_dict = {}
        for param_name, param_val in zip(param_keys, param_vals):
            if isinstance(param_val, list):
                param_val = ",".join([str(x) for x in param_val])
            arg_dict[param_name] = param_val
            args += "--{0}={1} ".format(param_name, param_val)

        # some ad hoc way to define rslt_dir_name, feel free to delete/comment out it
        rslt_dir_name = "fhn_SNR_0520/"
        #rslt_dir_name += arg_dict["datadir"].split("/")[-2]
        if arg_dict["smooth_obs"]:
            rslt_dir_name += "/" + ("RNN" if arg_dict["use_RNN"] else "attention")
        else:
            rslt_dir_name += "/" + ("FFBS" if arg_dict["FFBS"] else "filtering")
        
        args += "--rslt_dir_name={0}".format(rslt_dir_name)

        # create shell script
        with open(sh_name, "w") as f:
            f.write("#!/bin/bash\n")
            f.write("#$ -l mem={0}G,time={1}:: -S /bin/bash -N {2} -j y -cwd\n".format(mem, sh_time, task_name))
            f.write("cd {0}\n".format(conda_path))
            f.write("source activate {0}\n".format(env_name))
            f.write("cd {0}\n".format(execution_path))
            f.write("python {0} {1}\n".format(py_script_path, args))
            f.write("cd {0}\n".format(conda_path))
            f.write("source deactivate")

        # execute the shell script
        subprocess.Popen("qsub {0}".format(sh_name), shell=True)
        time.sleep(2)
