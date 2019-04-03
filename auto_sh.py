import subprocess
import itertools
import os
import time

# This is design for python 2.6
if __name__ == "__main__":

    mem = 8                        # memory in Gb
    sh_time = 48                    # time in hour
    task_name = "fhn_1D"            # name of the task
    conda_path = "/ifs/scratch/c2b2/ip_lab/zw2504/miniconda3/bin/"
    env_name = "vismc"
    execution_path = "/ifs/scratch/c2b2/ip_lab/zw2504/"
    py_script_path = "/ifs/scratch/c2b2/ip_lab/zw2504/VISMC/SMC_supreme/runner_flag.py"

    sh_name = "run"

    params_dict = {}

    # --------------------- training hyperparameters --------------------- #
    params_dict["Dx"] = [2]
    params_dict["Dy"] = [1]
    params_dict["Di"] = [1]
    params_dict["n_particles"] = [64]

    params_dict["batch_size"] = [1]
    params_dict["lr"] = [5e-4]
    params_dict["epoch"] = [600]
    params_dict["seed"] = [0]

    # --------------------- data set parameters --------------------- #
    params_dict["generateTrainingData"] = [False]

    params_dict["datadir"] = ["/ifs/scratch/c2b2/ip_lab/zw2504/VISMC/data/fhn/[1,0]_obs_cov_0.01/"]
    # "/ifs/scratch/c2b2/ip_lab/zw2504/VISMC/data/fitzhughnagumo/"
    # "/ifs/scratch/c2b2/ip_lab/zw2504/VISMC/data/fhn/[1,0]_obs_cov_0.01/"
    # "/ifs/scratch/c2b2/ip_lab/zw2504/VISMC/data/fhn/changing/"
    params_dict["datadict"] = ["datadict"]
    params_dict["isPython2"] = [False]

    params_dict["time"] = [200]
    params_dict["n_train"] = [200]
    params_dict["n_test"] = [40]

    # --------------------- model parameters --------------------- #
    # Feed-Forward Network (FFN)
    params_dict["q0_layers"] = [[64]]
    params_dict["q1_layers"] = [[64]]
    params_dict["q2_layers"] = [[64]]
    params_dict["f_layers"] = [[64]]
    params_dict["g_layers"] = [[64]]

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

    # Normalizing Flow (NF)
    params_dict["q1_flow_layers"] = [2]
    params_dict["f_flow_layers"] = [2]
    params_dict["flow_sample_num"] = [25]
    params_dict["flow_type"] = ["MAF"]

    # bidirectional RNN
    params_dict["y_smoother_Dhs"] = [[32]]
    params_dict["X0_smoother_Dhs"] = [[32]]

    # --------------------- SSM flags --------------------- #
    params_dict["use_bootstrap"] = [True]
    params_dict["q_uses_true_X"] = [False]
    params_dict["use_2_q"] = [True]
    params_dict["use_input"] = [False]
    params_dict["poisson_emission"] = [False]
    params_dict["flow_transition"] = [True]

    # --------------------- FFN flags --------------------- #
    params_dict["use_residual"] = [False]
    params_dict["output_cov"] = [False]
    params_dict["diag_cov"] = [False]
    params_dict["dropout_rate"] = [0.0]

    # --------------------- TFS flags --------------------- #
    params_dict["TFS"] = [True]
    params_dict["TFS_use_diff_q0"] = [True]

    # --------------------- FFBS flags --------------------- #
    params_dict["FFBS"] = [False]
    params_dict["smoothing_perc_factor"] = [1]
    params_dict["FFBS_to_learn"] = [True]

    # --------------------- smoother flags --------------------- #
    params_dict["smooth_obs"] = [False]
    params_dict["X0_use_separate_RNN"] = [True]
    params_dict["use_stack_rnn"] = [True]

    # --------------------- training flags --------------------- #
    params_dict["early_stop_patience"] = [200]
    params_dict["lr_reduce_patience"] = [40]
    params_dict["lr_reduce_factor"] = [0.7]
    params_dict["min_lr"] = [2e-5, 1e-5, 5e-6]
    params_dict["clip_norm"] = [10.0]

    # --------------------- printing and data saving params --------------------- #
    params_dict["print_freq"] = [5]
    params_dict["save_trajectory"] = [True]
    params_dict["save_y_hat"] = [True]

    params_dict["MSE_steps"] = [30]
    params_dict["saving_num"] = [30]

    params_dict["lattice_shape"] = [[25, 25]]  # [25, 25] or [10, 10, 5]
    params_dict["save_tensorboard"] = [False]
    params_dict["save_model"] = [False]

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
        rslt_dir_name = "fhn/"
        rslt_dir_name += arg_dict["datadir"].split("/")[-2]

        if arg_dict["TFS"]:
            rslt_dir_name += "/TFS"
        elif arg_dict["FFBS"]:
            rslt_dir_name += "/FFBS"
        elif arg_dict["smooth_obs"]:
            rslt_dir_name += "/RNN"
        else:
            rslt_dir_name += "/filtering"

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
