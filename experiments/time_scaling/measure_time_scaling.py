import subprocess
import os
import pickle


methods = ["frossl", "mmcr", "barlow", "simclr", "byol", "mocov2", "simsiam", "corinfomax", "vicreg"]

default_batch_size = 256
default_proj_dim = 1024
default_num_augs = 2

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
            if "loss_func" in line:
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
for method in methods:
    try:
        filename_prefix = f"{method}-{default_batch_size}-{default_proj_dim}-{default_num_augs}"
        batch_size=str(default_batch_size)
        proj_dim=str(default_proj_dim)
        num_augs=str(default_num_augs)

        # check if run exists in total_run_stats
        if filename_prefix in total_run_stats:
            continue

        # check if log file already exists
        if not os.path.exists(f"logs/fit-{filename_prefix}.txt"):
            # do training
            rc = subprocess.check_call(["./run.sh", method, filename_prefix, batch_size, proj_dim, num_augs])

        run_stats = parse_file(filename_prefix)
        total_run_stats[filename_prefix] = run_stats
        print(method, run_stats)
    except Exception as e:
        print(e)


print(total_run_stats)
# save total_run_stats to pickle
with open("logs/total_run_stats.pickle", "wb") as f:
    pickle.dump(total_run_stats, f)