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
    "ivne": "I-VNE",
    "swav": "SWAV"
}

ordererd_keys = [
    "simclr",
    "swav",
    "mocov2",
    "simsiam",
    "byol",
    "dino",
    "vicreg",
    "barlow",
    "wmse",
    "corinfomax",
    "ivne",
    "mmcr",
    "frossl"
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
    'dino': -1,          # 9wuauzyz
    'wmse': -1,          # 99jfb0l4
    'corinfomax': 405,   # pm2us0et
    'mmcr': 380,         # wk8vpqx6
    'mmcr-4view': 211,   # kr2o0q3a
    'mmcr-8view': 63     # 9skz9ogx
}

total_run_stats = {key:total_run_stats[key] for key in sorted(total_run_stats.keys(), key=lambda x: ordererd_keys.index(total_run_stats[x]["method"]))}

# make pandas dataframe for batch size sweep
batch_size_sweep_stats = {}
for key in total_run_stats:
    simple_key = method_name_to_formatted[total_run_stats[key]["method"]]
    if total_run_stats[key]["proj_dim"] == default_proj_dim and total_run_stats[key]["num_augs"] == default_num_augs:
        if simple_key not in batch_size_sweep_stats:
            batch_size_sweep_stats[simple_key] = {}

        if f"{total_run_stats[key]['loss_func_avg_time']:.3f}" == "0.000":
            formatted_loss_func_time = "1 ms"
        else:
            formatted_loss_func_time = f"{1000*float(total_run_stats[key]['loss_func_avg_time']):3.0f}".replace(' ', "\ ") + " ms"
        formatted_batch_avg_time = f"{1000*float(total_run_stats[key]['batch_avg_time']):3.0f}".replace(' ', "\ ")  + " ms"

        #formatted_entry = f"{formatted_loss_func_time} / {formatted_batch_avg_time} / {total_run_stats[key]['memory_gb']:.1f} GB"
        gbs_metric = int(total_run_stats[key]['memory_gb'] * total_run_stats[key]['batch_avg_time'] * 1000) # convert back from ms to s
    

        batch_size_sweep_stats[simple_key][total_run_stats[key]["batch_size"]] = formatted_batch_avg_time


df = pd.DataFrame.from_dict(batch_size_sweep_stats, orient='index')
print(df.to_latex(index=True, float_format="%.2f"))


# make pandas dataframe for dim size sweep
proj_dim_sweep_stats = {}
for key in total_run_stats:
    simple_key = method_name_to_formatted[total_run_stats[key]["method"]]
    if total_run_stats[key]["batch_size"] == default_batch_size and total_run_stats[key]["num_augs"] == default_num_augs:
        if simple_key not in proj_dim_sweep_stats:
            proj_dim_sweep_stats[simple_key] = {}

        if f"{total_run_stats[key]['loss_func_avg_time']:.3f}" == "0.000":
            formatted_loss_func_time = "1 ms"
        else:
            formatted_loss_func_time = f"{1000*float(total_run_stats[key]['loss_func_avg_time']):3.0f}".replace(' ', "\ ") + " ms"
        formatted_batch_avg_time = f"{1000*float(total_run_stats[key]['batch_avg_time']):3.0f}".replace(' ', "\ ")  + " ms"

        #formatted_entry = f"{formatted_loss_func_time} / {formatted_batch_avg_time} / {total_run_stats[key]['memory_gb']:.1f} GB"
        gbs_metric = int(total_run_stats[key]['memory_gb'] * total_run_stats[key]['batch_avg_time'] * 1000) # convert back from ms to s
    

        proj_dim_sweep_stats[simple_key][total_run_stats[key]["proj_dim"]] = formatted_batch_avg_time


df = pd.DataFrame.from_dict(proj_dim_sweep_stats, orient='index')
df = df.reindex(sorted(df.columns), axis=1)
print(df.to_latex(index=True, float_format="%.2f"))



# make pandas dataframe for dim size sweep
aug_sweep_stats = {}
for key in total_run_stats:
    simple_key = method_name_to_formatted[total_run_stats[key]["method"]]
    if total_run_stats[key]["num_augs"] != default_num_augs:

        if simple_key not in aug_sweep_stats:
            aug_sweep_stats[simple_key] = {}

        if f"{total_run_stats[key]['loss_func_avg_time']:.3f}" == "0.000":
            formatted_loss_func_time = "1 ms"
        else:
            formatted_loss_func_time = f"{1000*float(total_run_stats[key]['loss_func_avg_time']):3.0f}".replace(' ', "\ ") + " ms"
        formatted_batch_avg_time = f"{1000*float(total_run_stats[key]['batch_avg_time']):3.0f}".replace(' ', "\ ")  + " ms"

        #formatted_entry = f"{formatted_loss_func_time} / {formatted_batch_avg_time} / {total_run_stats[key]['memory_gb']:.1f} GB"
        gbs_metric = int(total_run_stats[key]['memory_gb'] * total_run_stats[key]['batch_avg_time'] * 1000) # convert back from ms to s

        aug_sweep_stats[simple_key][total_run_stats[key]["num_augs"]] = gbs_metric


df = pd.DataFrame.from_dict(aug_sweep_stats, orient='index')
df = df.reindex(sorted(df.columns), axis=1)
print(df.to_latex(index=True, float_format="%.2f"))

