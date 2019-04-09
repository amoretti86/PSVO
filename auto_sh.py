import argparse
import subprocess
import itertools
import os
import sys
import time
from datetools import addDateTime

# SBATCH/QSUB arguments
MEM = 32  # memory in Gb
SH_TIME = 24  # time in hour
TASK_NAME = "lorenz"  # name of the task
ENV_NAME = "tf"
RSLT_DIR = "fhn"
CLUSTER_COM = 'sbatch'
USER_EMAIL = "dh2832@columbia.edu"
CLUSTER = 'habanero'

# SBATCH arguments
ACCOUNT = 'stats'
NUM_CPU_CORES = 1

# TODO: Add remaining args for the cluster command
parser = argparse.ArgumentParser(description='Process arguments for sbatch')
parser.add_argument('--mem', default=MEM, metavar='M', type=int, nargs='+',
                    help='an integer for the accumulator')
parser.add_argument('--sum', dest='accumulate', action='store_const',
                    const=sum, default=max,
                    help='sum the integers (default: find the max)')

args = parser.parse_args()

lib_path = '/'.join(os.getcwd().split('/'))
sys.path.append(lib_path)


def run_batch():
    """
    """
#     execution_path = "/ifs/scratch/c2b2/ip_lab/zw2504/"
#     execution_path = lib_path + "/ifs/scratch/c2b2/ip_lab/zw2504/"

#     conda_path = "/ifs/scratch/c2b2/ip_lab/zw2504/miniconda3/bin/"
#     py_script_path = "/ifs/scratch/c2b2/ip_lab/zw2504/VISMC/SMC_supreme/runner_flag.py"
    py_script_path = lib_path + "/SMC_supreme/runner_flag.py"
    shell_script_name = "run.sh"

    params_dict = {}

    # --------------------- training hyperparameters --------------------- #
    params_dict["Dx"] = [3]
    params_dict["Dy"] = [1]
    params_dict["Di"] = [1]
    params_dict["n_particles"] = [500]

    params_dict["batch_size"] = [1]
    params_dict["lr"] = [1e-3]
    params_dict["epoch"] = [300]
    params_dict["seed"] = [0]

    # --------------------- data set parameters --------------------- #
    # generate synthetic data?
    params_dict["generateTrainingData"] = [False]

    # if reading data from file
    params_dict["datadir"] = ["/ifs/scratch/c2b2/ip_lab/zw2504/VISMC/data/lorenz/[1,0,0]_obs_cov_0.4/"]
    params_dict["datadict"] = ["datadict"]
    params_dict["isPython2"] = [False]

    # time, n_train and n_test will be overwritten if loading data from the file
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

    # bidirectional RNN
    params_dict["y_smoother_Dhs"] = [[16, 16]]
    params_dict["X0_smoother_Dhs"] = [[16, 16]]

    # Self-Attention encoder
    params_dict["num_hidden_layers"] = [4]
    params_dict["num_heads"] = [4]
    params_dict["hidden_size"] = [16]
    params_dict["filter_size"] = [16]
    params_dict["dropout_rate"] = [0.1]

    # --------------------- FFN flags --------------------- #
    params_dict["use_bootstrap"] = [True]
    params_dict["x_0_learnable"] = [True]
    params_dict["q_uses_true_X"] = [False]
    params_dict["use_residual"] = [False]
    params_dict["use_2_q"] = [True]
    params_dict["output_cov"] = [False]
    params_dict["use_input"] = [False]

    # --------------------- FFBS flags --------------------- #
    params_dict["FFBS"] = [False]
    params_dict["smoothing_perc_factor"] = [0]
    params_dict["FFBS_to_learn"] = [False]

    # --------------------- smoother flags --------------------- #
    params_dict["smooth_obs"] = [True]
    params_dict["use_RNN"] = [True]
    params_dict["X0_use_separate_RNN"] = [True]
    params_dict["use_stack_rnn"] = [True]

    # --------------------- training flags --------------------- #
    # stop training early if validation set does not improve
    params_dict["maxNumberNoImprovement"] = [200]
    params_dict["use_stop_gradient"] = [False]

    # --------------------- printing and data saving params --------------------- #
    params_dict["print_freq"] = [1]

    params_dict["store_res"] = [True]
    params_dict["MSE_steps"] = [10]

    # how many trajectories to draw in quiver plot
    params_dict["quiver_traj_num"] = [30]
    params_dict["lattice_shape"] = [[10, 10, 3]]  # [25, 25] or [10, 10, 3]

    params_dict["saving_num"] = [30]

    params_dict["save_tensorboard"] = [False]
    params_dict["save_model"] = [False]
    params_dict["save_freq"] = [10]

    # --------------------- parameters part ends --------------------- # DANI:
    # This is dangerous, I think it only works because python 3.5+ preserves the
    # dict ordering. This is not the case for previous pythons!
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
        rslt_dir_name = RSLT_DIR
        rslt_dir_name += arg_dict["datadir"].split("/")[-2]
        if arg_dict["smooth_obs"]:
            rslt_dir_name += "/" + ("RNN" if arg_dict["use_RNN"] else "attention")
        else:
            rslt_dir_name += "/" + ("FFBS" if arg_dict["FFBS"] else "filtering")

        args += "--rslt_dir_name={0}".format(rslt_dir_name)

        # create shell script
        if CLUSTER_COM == 'qsub':
            with open(shell_script_name, "w") as f:
                f.write("#!/bin/bash\n")
                f.write("#$ -l MEM={0}G,time={1}:: -S /bin/bash "
                        "-N {2} -j y -cwd\n".format(MEM, SH_TIME, TASK_NAME))
    #             f.write("cd {0}\n".format(conda_path)) # conda binaries assumed to be on $PATH
                f.write("source activate {0}\n".format(ENV_NAME))
    #             f.write("cd {0}\n".format(execution_path))
                f.write("python {0} {1}\n".format(py_script_path, args))
    #             f.write("cd {0}\n".format(conda_path))
                f.write("source deactivate")
        elif CLUSTER_COM == 'sbatch':
            with open(shell_script_name, "w") as f:
                rslt_dir_path = lib_path + '/rslts/' + RSLT_DIR + '/' + addDateTime()
                print("rslt_dir_path", rslt_dir_path)
                if not os.path.exists(rslt_dir_path):
                    os.makedirs(rslt_dir_path)
                f.write("#!/bin/sh\n\n")
                f.write("#SBATCH --account={0}\n".format(ACCOUNT))
                f.write("#SBATCH --job-name={0}\n".format(TASK_NAME))
                f.write("#SBATCH -c {}\n".format(NUM_CPU_CORES))
                f.write("#SBATCH --time={0}\n".format(str(SH_TIME) + ":00:00"))
                f.write("#SBATCH --mem-per-cpu={0}gb\n".format(MEM))
                f.write("#SBATCH --workdir={0}\n".format(lib_path))
                f.write("#SBATCH --error={0}/e_%j.out\n".format(rslt_dir_path))
                f.write("#SBATCH --output={0}/o_%j.out\n".format(rslt_dir_path))
                f.write("#SBATCH --mail-type=BEGIN\n")
                f.write("#SBATCH --mail-type=END\n")
                f.write("#SBATCH --mail-type=FAIL\n")
                f.write("#SBATCH --mail-user={0}\n".format(USER_EMAIL))

                if CLUSTER == 'habanero':
                    f.write("module load anaconda\n")
#                 f.write("\nsource activate {0}\n".format(ENV_NAME))
                    f.write("python {0} {1}\n".format(py_script_path, args))
#                 f.write("source deactivate")
        else:
            raise ValueError("Cluster command {0} not recognized [allowed ones = "
                             'qsub, sbatch'"]".format(CLUSTER_COM))

        # execute the shell script
#         subprocess.Popen(CLUSTER_COM + " {0}".format(shell_script_name), shell=True)
#         time.sleep(2)


# Fly babe!
run_batch()
