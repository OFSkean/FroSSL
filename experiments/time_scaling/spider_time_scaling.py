import matplotlib.pyplot as plt
import os
import pickle
import pandas as pd
from spider_utils import make_spider

default_batch_size = 256
default_proj_dim = 1024
default_num_augs = 2

import plotly.graph_objects as go

categories = ['processing cost','mechanical properties','chemical stability',
              'thermal stability', 'device integration']



# load total_run_stats if pkl exists
if os.path.exists("logs/total_run_stats.pickle"):
    with open("logs/total_run_stats.pickle", "rb") as f:
        total_run_stats = pickle.load(f)
else:
    raise Exception("total_run_stats.pickle does not exist")

# iterate through total_run_stats and add info keys
for key in total_run_stats:
    total_run_stats[key]["method"] = key.split("-")[0]
    total_run_stats[key]["batch_size"] = int(key.split("-")[1])
    total_run_stats[key]["proj_dim"] = int(key.split("-")[2])
    total_run_stats[key]["num_augs"] = int(key.split("-")[3])
    total_run_stats[key]["memory_gb"] = float(total_run_stats[key]["memory_gb"])
    total_run_stats[key]["loss_func_avg_time"] = float(total_run_stats[key]["loss_func_avg_time"])
    total_run_stats[key]["batch_avg_time"] = float(total_run_stats[key]["batch_avg_time"])

method_name_to_formatted = {
    "frossl": "FroSSL",
    "simsiam": "SimSiam",
    "barlow": "Barlow Twins",
    "corinfomax": "CorInfoMax",
    "byol": "BYOL",
    "mocov2": "MoCoV2",
    "simclr": "SimCLR",
    "mmcr": "MMCR",
    "vicreg": "VICReg",
    "mmcr-4view": "MMCR-4view",
    "mmcr-8view": "MMCR-8view",
    "frossl-4view": "FroSSL-4view",
    "frossl-8view": "FroSSL-8view"
}

method_name_to_hyperparameters = {
    "frossl": 1,
    "simsiam": 2,
    "barlow": 1,
    "corinfomax": 5,
    "byol": 3,
    "mocov2": 4,
    "simclr": 1,
    "mmcr": 0,
    "vicreg": 3,
    "mmcr-4view": 0,
    "mmcr-8view": 0,
    "frossl-4view": 1,
    "frossl-8view": 1
}

ordererd_keys = [
    "simclr",
    "frossl",
    "swav",
    "mocov2",
    "mmcr",
    "simsiam",
    "byol",
    "frossl-4view",
    "dino",
    "vicreg",
    "mmcr-4view",
    "barlow",
    "frossl-8view",
    "corinfomax",
    "mmcr-8view",
    "wmse",
]

convergence_to_80_epochs = {
    'frossl': 290,       # rynz0sgg
    'frossl-4view': 144, # inkhztwt
    'frossl-8view': 51,  # w1k8auv6
    'vicreg': 360,       # yhr5ev1j
    'simclr': 347,       # w3i1c17d
    'barlow': 370,       # 4jw5419m
    'simsiam': 145,      # zalllgem
    'mocov2': 180,       # h1om782q
    'swav': 240,         # pu2n7ocv
    'byol': 187,         # 74su6fg3
    'dino': 1000000,     # 9wuauzyz
    'wmse': 1000000,     # 99jfb0l4
    'corinfomax': 405,   # pm2us0et
    'mmcr': 380,         # wk8vpqx6
    'mmcr-4view': 211,   # kr2o0q3a
    'mmcr-8view': 63     # 9skz9ogx
}

method_to_accuracies = {
    'frossl': 87.3,       # rynz0sgg
    'frossl-4view': 90.0, # inkhztwt
    'frossl-8view': 90.9,  # w1k8auv6
    'vicreg': 85.9,       # yhr5ev1j
    'simclr': 85.9,       # w3i1c17d
    'barlow': 85.0,       # 4jw5419m
    'simsiam': 88.5,      # zalllgem
    'mocov2': 83.2,       # h1om782q
    'swav': 82.6,         # pu2n7ocv
    'byol': 88.7,         # 74su6fg3
    'dino': 78.9,     # 9wuauzyz
    'wmse': 72.4,     # 99jfb0l4
    'corinfomax': 83.1,   # pm2us0et
    'mmcr': 84.3,         # wk8vpqx6
    'mmcr-4view': 88.2,   # kr2o0q3a
    'mmcr-8view': 90.3     # 9skz9ogx
}

# get run stats for default batch size and default proj dim
relevant_run_stats = {key:total_run_stats[key] for key in total_run_stats if total_run_stats[key]["batch_size"] == default_batch_size and total_run_stats[key]["proj_dim"] == default_proj_dim}
relevant_run_stats.pop("simclr-256-1024-4") # remove multiview
relevant_run_stats.pop("simclr-256-1024-8") # remove multiview
relevant_run_stats.pop("simsiam-256-1024-2") # remove duplicate

# adjust method names for multiview methods
for key in relevant_run_stats:
    if relevant_run_stats[key]["method"] == "mmcr":
        if relevant_run_stats[key]["num_augs"] == 2:
            relevant_run_stats[key]["method"] = "mmcr"
        elif relevant_run_stats[key]["num_augs"] == 4:
            relevant_run_stats[key]["method"] = "mmcr-4view"
        elif relevant_run_stats[key]["num_augs"] == 8:
            relevant_run_stats[key]["method"] = "mmcr-8view"

    elif relevant_run_stats[key]["method"] == "frossl":
        if relevant_run_stats[key]["num_augs"] == 2:
            relevant_run_stats[key]["method"] = "frossl"
        elif relevant_run_stats[key]["num_augs"] == 4:
            relevant_run_stats[key]["method"] = "frossl-4view"
        elif relevant_run_stats[key]["num_augs"] == 8:
            relevant_run_stats[key]["method"] = "frossl-8view"


relevant_run_stats = {key:relevant_run_stats[key] for key in sorted(relevant_run_stats.keys(), key=lambda x: ordererd_keys.index(relevant_run_stats[x]["method"]))}

# look through run stats for smallest batch time for default batch size and default proj dim
min_batch_time = min([relevant_run_stats[key]["batch_avg_time"] for key in relevant_run_stats])
min_memory = min([relevant_run_stats[key]["memory_gb"] for key in relevant_run_stats])
min_epochs = min([convergence_to_80_epochs[relevant_run_stats[key]["method"]] for key in relevant_run_stats])

max_batch_time = max([relevant_run_stats[key]["batch_avg_time"] for key in relevant_run_stats])
max_memory = max([relevant_run_stats[key]["memory_gb"] for key in relevant_run_stats])
max_epochs = max([convergence_to_80_epochs[relevant_run_stats[key]["method"]] for key in relevant_run_stats])

# corner points of the regions. defined as ratio of batch time to min batch time, ratio of memory to min memory, ratio of epochs to min epochs
memorys = []
times = []
wall_times = []
formatted_names = []
epochs = []
hyperparams = []
accuracies = []
for key in relevant_run_stats:
    wall_time =  410*relevant_run_stats[key]["batch_avg_time"]*convergence_to_80_epochs[relevant_run_stats[key]["method"]]/3600

    memorys.append(relevant_run_stats[key]["memory_gb"])
    times.append(relevant_run_stats[key]["batch_avg_time"])
    wall_times.append(wall_time)
    formatted_names.append(method_name_to_formatted[relevant_run_stats[key]["method"]])
    epochs.append(convergence_to_80_epochs[relevant_run_stats[key]["method"]])
    hyperparams.append(method_name_to_hyperparameters[relevant_run_stats[key]["method"]])
    accuracies.append(method_to_accuracies[relevant_run_stats[key]["method"]])

# normalize values to 1
memorys = [x / 8 for x in memorys]
times = [x / 0.2 for x in times]
wall_times = [x / 5 for x in wall_times]
epochs = [x / 500 for x in epochs]
hyperparams = [x / 5 for x in hyperparams]
accuracies = [(x - 70) / 30 for x in accuracies]

df = pd.DataFrame({
    'group': formatted_names,
    'memory': memorys,
    'time': times,
    'wall_time': wall_times,
    'epochs': epochs,
    'hyperparams': hyperparams,
    'accuracy': accuracies
})

my_dpi=96
height = 1500
width = 3500
numrows = 2
plt.figure(figsize=(width/my_dpi, height/my_dpi), dpi=my_dpi)
 
# Create a color palette:
my_palette = plt.cm.get_cmap("Set2", len(df.index))
 
# Loop to plot
for row in range(0, len(df.index)):
    make_spider(df, row=row, title=df['group'][row], color=my_palette(row))

plt.tight_layout()
plt.savefig("spider.png")