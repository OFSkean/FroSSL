import subprocess

batch_sizes = [256]
proj_dim = [1024]
num_augs = [2]

rc = subprocess.check_call(["./run.sh", "frossl", "frossl-256-1024", "256", "1024", "2"])