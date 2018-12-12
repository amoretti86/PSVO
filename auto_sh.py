import subprocess
import itertools
import os
import time

if __name__ == "__main__":

    mem = 32                        # memory in Gb
    sh_time = 72                    # time in hour
    task_name = "FHN"               # name of the task
    conda_path = "/ifs/scratch/c2b2/ip_lab/zw2504/miniconda3/bin/"
    env_name = "vismc"
    execution_path = "/ifs/scratch/c2b2/ip_lab/zw2504/"
    py_script_path = "/ifs/scratch/c2b2/ip_lab/zw2504/VISMC/SMC_supreme/runner.py"

    sh_name = "run.sh"

    params_dict = {}
    params_dict["Dx"] = [2]
    params_dict["Dy"] = [1]
    params_dict["n_particles"] = [500]

    params_dict["batch_size"] = [5]
    params_dict["lr"] = [1e-3]
    params_dict["epoch"] = [300]
    params_dict["seed"] = [0, 1, 2, 3, 4]

    # time, n_train and n_test will be overwritten if loading data from the file
    params_dict["time"] = [200]
    params_dict["n_train"] = [200]
    params_dict["n_test"] = [40]

    # Define encoder and decoder network architectures
    params_dict["q_train_layers"] = [[50]]
    params_dict["f_train_layers"] = [[50]]
    params_dict["g_train_layers"] = [[50]]

    # do q and f use the same network?
    params_dict["use_bootstrap"] = [True]

    # if q takes y_t as input
    # if is_bootstrap, q_takes_y will be overwritten as False
    params_dict["q_takes_y"] = [False]

    # should q use true_X to sample? (useful for debugging)
    params_dict["q_uses_true_X"] = [False]

    # term to weight the added contribution of the MSE to the cost
    params_dict["loss_beta"] = [0.0, 0.2, 0.25]

    # stop training early if validation set does not improve
    params_dict["maxNumberNoImprovement"] = [100]

    # generate synthetic data?
    params_dict["generateTrainingData"] = [False]

    # if reading data from file
    params_dict["datadir"] = ["C:/Users/admin/Desktop/research/code/VISMC/data/lorenz/[1,0,0]_obs_cov_0.4/"]
    params_dict["datadict"] = ["data.p"]
    params_dict["isPython2"] = [False]
    params_dict["rslt_dir_name"] = ["FN_1D_obs"]

    param_keys = list(params_dict.keys())
    param_values = list(params_dict.values())
    param_vals_permutation = list(itertools.product(*param_values))
    for param_vals in param_vals_permutation:
        # create args
        args = ""
        for param_name, param_val in zip(param_keys, param_vals):
            args += "--{0} {1} ".format(param_name, param_val)

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
        time.sleep(5)
