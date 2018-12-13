import subprocess
import itertools
import os
import time

# This is design for python 2.6
if __name__ == "__main__":

    mem = 32                        # memory in Gb
    sh_time = 72                    # time in hour
    task_name = "lorenz"               # name of the task
    conda_path = "/ifs/scratch/c2b2/ip_lab/zw2504/miniconda3/bin/"
    env_name = "vismc"
    execution_path = "/ifs/scratch/c2b2/ip_lab/zw2504/"
    py_script_path = "/ifs/scratch/c2b2/ip_lab/zw2504/VISMC/SMC_supreme/runner_flag.py"

    sh_name = "run.sh"

    params_dict = {}

    # --------------------- training hyperparameters --------------------- #
    params_dict["Dx"] = [3]
    params_dict["Dy"] = [1]
    params_dict["n_particles"] = [500]

    params_dict["batch_size"] = [1]
    params_dict["lr"] = [1e-3, 3e-4]
    params_dict["epoch"] = [300]
    params_dict["seed"] = [0]

    # --------------------- data set parameters --------------------- #
    # generate synthetic data?
    params_dict["generateTrainingData"] = [False]

    # if reading data from file
    params_dict["datadir"] = ["/ifs/scratch/c2b2/ip_lab/zw2504/VISMC/data/lorenz/[1,0,0]_obs_cov_0.4"]
    params_dict["datadict"] = ["datadict"]
    params_dict["isPython2"] = [False]

    # time, n_train and n_test will be overwritten if loading data from the file
    params_dict["time"] = [200]
    params_dict["n_train"] = [200]
    params_dict["n_test"] = [40]

    # --------------------- model parameters --------------------- #
    # Define encoder and decoder network architectures
    params_dict["q_train_layers"] = [[50]]
    params_dict["f_train_layers"] = [[50]]
    params_dict["g_train_layers"] = [[50]]

    params_dict["q_sigma_init"] = [5]
    params_dict["f_sigma_init"] = [5]
    params_dict["g_sigma_init"] = [5]
    params_dict["q_sigma_min"] = [1]
    params_dict["f_sigma_min"] = [1]
    params_dict["g_sigma_min"] = [1]

    # do q and f use the same network?
    params_dict["use_bootstrap"] = [True]

    # if q takes y_t as input
    # if is_bootstrap, q_takes_y will be overwritten as False
    params_dict["q_takes_y"] = [False]

    # should q use true_X to sample? (useful for debugging)
    params_dict["q_uses_true_X"] = [False]

    # term to weight the added contribution of the MSE to the cost
    params_dict["loss_beta"] = [0.0]

    # stop training early if validation set does not improve
    params_dict["maxNumberNoImprovement"] = [5]

    params_dict["x_0_learnable"] = [True, False]
    params_dict["use_residual"] = [True, False]
    params_dict["output_cov"] = [True, False]

    # --------------------- printing and data saving params --------------------- #
    params_dict["print_freq"] = [5]

    params_dict["store_res"] = [True]
    # params_dict["rslt_dir_name"] = "lorenz_1D"
    params_dict["MSE_steps"] = [10]

    # how many trajectories to draw in quiver plot
    params_dict["quiver_traj_num"] = [5]
    params_dict["lattice_shape"] = [[10, 10, 3]]  # [25, 25] or [10, 10, 3]

    params_dict["saving_num"] = [10]

    params_dict["save_tensorboard"] = [False]
    params_dict["save_model"] = [False]
    params_dict["save_freq"] = [10]

    # --------------------- parameters part ends --------------------- #
    param_keys = list(params_dict.keys())
    param_values = list(params_dict.values())
    param_vals_permutation = list(itertools.product(*param_values))

    for param_vals in param_vals_permutation:
        args = ""
        arg_dict = {}
        for param_name, param_val in zip(param_keys, param_vals):
            if isinstance(param_val, list):
                param_val = ",".join([str(x) for x in param_val])
            arg_dict[param_name] = param_val
            args += "--{0}={1} ".format(param_name, param_val)

        # some ad hoc way to define rslt_dir_name, feel free to delete/comment out it
        args += "--rslt_dir_name {0}".format("/".join(arg_dict["datadir"].split("/")[-3:]))

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
        time.sleep(0.25)
