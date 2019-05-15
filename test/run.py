import os

import autograd.numpy as np

import utils.load


def match_kwargs_and_df(list_of_kwargs, df):

    for kwargs, (ii, row) in zip(list_of_kwargs, df.iterrows()):

        # regularization
        if "regularizer" in kwargs.keys():
            assert kwargs["regularizer"] == row.regularizer_str

        # network

        if "include_biases" in kwargs.keys():
            if kwargs["include_biases"] is None:
                assert not row.has_biases
            else:
                assert row.has_biases

        # optimizer

        if "optimizer" in kwargs.keys():
            if kwargs["optimizer"] is not None:
                assert kwargs["optimizer"] == row.optimizer_str

        if "optimizer_lr" in kwargs.keys():
            if kwargs["optimizer_lr"] is not None:
                assert kwargs["optimizer_lr"] == row.optimizer_kwargs["lr"]

        if "optimizer_momentum" in kwargs.keys():
            if kwargs["optimizer_momentum"] is not None:
                assert kwargs["optimizer_momentum"] == row.optimizer_kwargs["momentum"]


def recreation_test(row, experiment_type="optimization"):

    if experiment_type == "optimization":
        data_json_path, network_json_path = row.data_json, row.network_json,
        optimizer_json_path = row.optimizer_json
        _, _, experiment = utils.load.from_paths(
            data_json_path, network_json_path, optimizer_json_path)

        thetas = get_thetas(os.path.dirname(data_json_path))

    elif experiment_type == "critfinder":
        experiment_json_path = row.experiment_json
        experiment_json = utils.load.open_json(experiment_json_path)
        optimization_path = experiment_json["optimization_path"]

        data_json_path = os.path.join(optimization_path, "data.json")
        network_json_path = os.path.join(optimization_path, "network.json")

        finder_json_path = row.finder_json

        _, _, experiment = utils.load.from_paths(
            data_json_path, network_json_path, finder_json_path,
            experiment_type=experiment_type)

        thetas = get_thetas(os.path.dirname(finder_json_path))

    experiment_output = experiment.run(thetas[0], len(thetas) - 1)

    if experiment_type == "optimization":
        recreated_thetas = experiment_output
    elif experiment_type == "critfinder":
        recreated_thetas = experiment.runs[-1]["theta"]

    if not np.array_equal(thetas, recreated_thetas):
        return (False, [thetas, recreated_thetas])
    else:
        return (True, None)


def get_thetas(optimization_folder):
    thetas = np.load(os.path.join(optimization_folder, "trajectories", "0000.npz"))["theta"]
    return thetas
