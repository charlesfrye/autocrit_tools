"""Utilities for building dataframes with metadata and paths for experiments and critical points.
"""
import os

import autograd.numpy as np
import pandas as pd

from . import load
from .path import ExperimentPaths


def construct_experiments_df(experiments_path, experiment_type="optimization"):
    """Constructs a dataframe containing configuration information for
    all experiments in a given folder, either optimization or critfinder.
    """
    experiments_path_elems = [os.path.join(experiments_path, elem)
                              for elem in os.listdir(experiments_path)]

    experiment_paths = [elem for elem in experiments_path_elems
                        if os.path.isdir(elem) and is_experiment_dir(elem)]

    experiment_IDs = [os.path.basename(experiment_path)
                      for experiment_path in experiment_paths]

    rows = [construct_experiment_row(experiment_path, experiment_type=experiment_type)
            for experiment_path in experiment_paths]

    return pd.DataFrame(data=rows, index=experiment_IDs)


def construct_experiment_row(experiment_path, experiment_type="optimization"):
    if experiment_type not in ["optimization", "critfinder"]:
        raise NotImplementedError("experiment_type {} not understood"
                                  .format(experiment_type))

    if experiment_type == "optimization":
        paths = ExperimentPaths.from_optimizer_dir(experiment_path)
    else:
        paths = ExperimentPaths.from_critfinder_dir(experiment_path)

    json_names, json_paths = zip(*[(name, val)
                                 for name, val in paths.jsons.items() if val is not None])

    json_dicts = [load.open_json(json_path) for json_path in json_paths]

    experiment_row = {}
    for json_dict, json_name in zip(json_dicts, json_names):
        json_dict = {key + "_" + json_name: val for key, val in json_dict.items()}
        experiment_row.update(json_dict)

    for json_name, json_path in zip(json_names, json_paths):
        experiment_row[json_name + "_json"] = json_path

    dir_names, dir_paths = zip(*[(name, val)
                               for name, val in paths.directories.items() if val is not None])

    for dir_name, dir_path in zip(dir_names, dir_paths):
        experiment_row[dir_name + "_dir"] = dir_path

    experiment_row["data_path"] = paths.data

    return experiment_row


def is_experiment_dir(dir):
    """quick and dirty check"""
    return any([elem.endswith("finder.json") or elem.endswith("optimizer.json")
                for elem in os.listdir(dir)])


def reconstruct_from_row(experiment_row, experiment_type="optimization"):

    if experiment_type not in ["optimization", "critfinder"]:
        raise NotImplementedError("experiment_type {} not understood"
                                  .format(experiment_type))

    if experiment_type == "critfinder":
        paths = ExperimentPaths.from_finder_dir(experiment_row.finder_dir)
        experiment_json_path = paths.finder
    else:
        paths = ExperimentPaths.from_optimizer_dir(experiment_row.optimizer_dir)
        experiment_json_path = paths.optimizer

    data_path = paths.data
    network_json_path = paths.network

    data, network, experiment = load.from_paths(
        data_path, network_json_path, experiment_json_path,
        experiment_type=experiment_type)

    return data, network, experiment


def construct_cp_df(critfinder_row):

    finder_kwargs = critfinder_row.finder_kwargs_finder

    finder_dir = os.path.dirname(critfinder_row.finder_json)
    paths = ExperimentPaths.from_critfinder_dir(finder_dir)

    output_dir = paths.finder_out_dir

    output_npzs = [np.load(str(output_dir / elem))
                   for elem in os.listdir(output_dir)
                   if elem.endswith("npz")]

    row_dictionaries = []
    for output_npz in output_npzs:
        row_dictionary = {}

        row_dictionary.update(finder_kwargs)

        if "theta" in output_npz.keys():
            row_dictionary["thetas"] = output_npz["theta"]
            row_dictionary["run_length"] = len(row_dictionary["thetas"])
            if row_dictionary["run_length"] > 0:
                row_dictionary["final_theta"] = row_dictionary["thetas"][-1]

        if "f_theta" in output_npz.keys():
            row_dictionary["losses"] = output_npz["f_theta"]
            if len(row_dictionary["losses"]) > 0:
                row_dictionary["final_loss"] = row_dictionary["losses"][-1]

        if "g_theta" in output_npz.keys():
            if len(output_npz["g_theta"]) > 0:
                row_dictionary["squared_grad_norms"] = 2 * output_npz["g_theta"]
                row_dictionary["final_squared_grad_norm"] = row_dictionary["squared_grad_norms"][-1]

        if "alpha" in output_npz.keys():
            row_dictionary["alphas"] = output_npz["alpha"]

        if "pure_accepted" in output_npz.keys():
            row_dictionary["pure_accepted"] = output_npz["pure_accepted"]

        row_dictionaries.append(row_dictionary)

    return pd.DataFrame(row_dictionaries)
