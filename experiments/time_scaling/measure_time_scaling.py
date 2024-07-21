import subprocess
import os
import pickle
import itertools

methods = ["frossl", "mmcr", "barlow", "simclr", "ivne", "byol", "mocov2", "simsiam", "corinfomax", "vicreg", "swav"]
#methods = ["frossl", "mmcr", "ivne", "corinfomax"]
multiaug_compatible_methods = ["frossl", "mmcr"]

default_batch_size = 1024
default_proj_dim = 1024
default_num_augs = 2

batch_sizes_list = [1024]
proj_dims_list = [1024]
augs_list = [2, 4, 8]

method_batch_sweep = list(itertools.product(methods, batch_sizes_list, [default_proj_dim], [default_num_augs]))
method_proj_sweep = list(itertools.product(methods, [default_batch_size], proj_dims_list, [default_num_augs]))
method_aug_sweep = list(itertools.product(multiaug_compatible_methods, [default_batch_size], [default_proj_dim], augs_list))
combos = method_batch_sweep + method_proj_sweep + method_aug_sweep

def parse_file(filename):
    # parse through log file for memory and time details
    with open(f"logs/fit-{filename}.txt", "r") as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            line = line.strip()

            # maximum memory used
            if "Memory" and "GB" in line:
                memory_gb = line.split(" ")[-2]

            # loss function time
            # corinfomax uses CorInfoMax_Loss class, not func
            if "loss_func" in line or 'corinfomax.py:36(forward)' in line:
                parsed_line = [x for x in line.strip().split(" ") if x != '']
                loss_func_avg_time = parsed_line[4]

            # total time for a batch (includes loss function time)
            # automatic.py:157(run) is main loop for batch
            if "automatic.py:157(run)" in line:
                parsed_line = [x for x in line.strip().split(" ") if x != '']
                batch_avg_time = parsed_line[4]

    run_stats = {
        "memory_gb": memory_gb,
        "loss_func_avg_time": loss_func_avg_time,
        "batch_avg_time": batch_avg_time
    }

    return run_stats


# load total_run_stats if pkl exists
if os.path.exists("logs/total_run_stats.pickle"):
    with open("logs/total_run_stats.pickle", "rb") as f:
        total_run_stats = pickle.load(f)
else:
    total_run_stats = {}

# train each method with default hyperparameters
for method_combo in combos:
    method, batch_size, proj_dim, num_augs = method_combo

    try:
        filename_prefix = f"{method}-{batch_size}-{proj_dim}-{num_augs}"
        batch_size=str(batch_size)
        proj_dim=str(proj_dim)
        num_augs=str(num_augs)

        # check if run exists in total_run_stats
        # if filename_prefix in total_run_stats:
        #     continue

        # check if log file already exists
        if not os.path.exists(f"logs/fit-{filename_prefix}.txt"):
            # do training
            rc = subprocess.check_call(["./run.sh", method, filename_prefix, batch_size, proj_dim, num_augs])

        run_stats = parse_file(filename_prefix)
        total_run_stats[filename_prefix] = run_stats
        print(method, filename_prefix, run_stats)
    except Exception as e:
        print(e)


print(total_run_stats)
# save total_run_stats to pickle
with open("logs/total_run_stats.pickle", "wb") as f:
    pickle.dump(total_run_stats, f)