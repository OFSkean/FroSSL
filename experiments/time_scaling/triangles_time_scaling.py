from triangles_utils import fill_region
import ternary
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

import os
import pickle
import pandas as pd

default_batch_size = 256
default_proj_dim = 1024
default_num_augs = 2

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
regions = []
labels = []
memorys = []
times = []
wall_times = []
eps = 0.03
for key in relevant_run_stats:
    #batch_time_ratio = max(1 - min_batch_time /  relevant_run_stats[key]["batch_avg_time"], eps)
    #batch_time_ratio =  max(eps, (relevant_run_stats[key]["batch_avg_time"] - min_batch_time) / (max_batch_time))
    batch_time_ratio =  (relevant_run_stats[key]["batch_avg_time"] ) / (max_batch_time)
    #memory_ratio = max(1 - min_memory /  relevant_run_stats[key]["memory_gb"], eps)
    #memory_ratio =  max(eps, (relevant_run_stats[key]["memory_gb"] - min_memory) / (max_memory - min_memory))
    memory_ratio =  (relevant_run_stats[key]["memory_gb"] ) / (max_memory)
    #epochs_ratio =  (convergence_to_80_epochs[relevant_run_stats[key]["method"]] - min_epochs) / (max_epochs - min_epochs)
    epochs_ratio =  (convergence_to_80_epochs[relevant_run_stats[key]["method"]] ) / (max_epochs)
    
    # graph from center
    batch_time_ratio = batch_time_ratio * (2/3) + (1/3)
    memory_ratio = memory_ratio * (2/3) + (1/3)
    epochs_ratio = epochs_ratio * (2/3) + (1/3)
    wall_time =  410*relevant_run_stats[key]["batch_avg_time"]*convergence_to_80_epochs[relevant_run_stats[key]["method"]]

    # black magic
    batch_time_offset = (1 - batch_time_ratio)/2
    memory_offset = (1 - memory_ratio)/2
    epochs_offset = (1 - epochs_ratio)/2

    regions.append([(memory_ratio, memory_offset, memory_offset), 
                     (epochs_offset, epochs_ratio, epochs_offset), 
                     (batch_time_offset, batch_time_offset, batch_time_ratio)])
    labels.append(relevant_run_stats[key]["method"])
    memorys.append(relevant_run_stats[key]["memory_gb"])
    times.append(relevant_run_stats[key]["batch_avg_time"])
    wall_times.append(wall_time)
    


eps=0.01
center_point = [(1/3+eps, 1/3+eps, 1/3-2*eps), (1/3+eps, 1/3-2*eps, 1/3+eps), (1/3-2*eps, 1/3+eps, 1/3+eps)]

cmap = cm.plasma
colorlist = []
for i in np.arange(0,1,1/len(regions)):
    colorlist.append(cmap(i))

scale = 1
fontsize = 30
figure, axis = plt.subplots(2, 6)
figure.set_size_inches(35, 10.5)

for idx, r in enumerate(regions):
    cur_ax = axis[idx%2][idx//2]
    t_figure, t_ax = ternary.figure(ax=cur_ax, scale=scale)
    t_ax.gridlines(color="grey", multiple=0.1)
    fill_region(t_ax.get_axes(), color=colorlist[3], points=r, alpha=.7)
    fill_region(t_ax.get_axes(), color='black', points=center_point, alpha=.7)

   
    label = method_name_to_formatted[labels[idx]]
    # add label below cur_ax
    cur_ax.text(0.5, -0.1, label, 
                size=fontsize, ha="center", transform=cur_ax.transAxes)
    
    wall_time_hours = wall_times[idx]/(3600)

    # add label at top corner for epochs
    epochs_label = f"{convergence_to_80_epochs[labels[idx]]}"
    epoch_weight = 'bold' if convergence_to_80_epochs[labels[idx]] <= min_epochs*1.01 else 'regular'
    if idx == 0:
        epochs_label = f"{epochs_label} epochs"
    epochs_label = f"{epochs_label} $\\approx$ {wall_time_hours:.1f} hours"
    cur_ax.text(0.5, 1.05, epochs_label, weight=epoch_weight,
                size=fontsize, ha="center", transform=cur_ax.transAxes)
    
    # add label at bottom right corner for memory
    memory_label = f"{memorys[idx]:.1f}"
    memory_weight = 'bold' if memorys[idx] <= min_memory*1.05 else 'regular'
    if idx == 0:
        memory_label = f"{memory_label} GB"
        cur_ax.text(1.15, -0.05, memory_label, weight=memory_weight,
            size=fontsize, ha="center", transform=cur_ax.transAxes)
    else:
        cur_ax.text(1.05, 0.15, memory_label, weight=memory_weight,
                    size=fontsize, ha="center", transform=cur_ax.transAxes)

    # add label at bottom left corner for time
    time_label = f"{times[idx]*1000:.0f}"
    time_weight = 'bold' if times[idx] <= min_batch_time*1.05 else 'regular'
    if idx == 0:
        time_label = f"{time_label} ms"
        cur_ax.text(-0.05, -0.05, time_label, weight=time_weight,
                    size=fontsize, ha="center", transform=cur_ax.transAxes)
    else:
        cur_ax.text(-0.05, 0.15, time_label, weight=time_weight,
            size=fontsize, ha="center", transform=cur_ax.transAxes)
    
    
    # Final plot formatting
    t_ax.boundary(linewidth=2)
    t_ax.get_axes().axis('off')
    t_ax.clear_matplotlib_ticks()
    t_ax.get_axes().set_aspect(1)
    t_ax._redraw_labels()
    
figure.tight_layout()
figure.savefig('ssl_method_triangles.png')