import json
import pickle

import autograd.numpy as np

import autocrit.experiments
import autocrit.nn.networks


def fetch_data(data_path):
    arrays = np.load(data_path)

    if len(arrays) < 2:
        data = (arrays["xs"], arrays["xs"])
    elif len(arrays) == 2:
        data = (arrays["xs"], arrays["ys"])
    else:
        raise ValueError("loaded data.npz has too many arrays")

    return data


def open_json(jfn):
    with open(jfn) as jf:
        json_data = json.load(jf)
    return json_data


def from_paths(data_path, network_json_path, experiment_json_path,
               experiment_type="optimization"):
    data = fetch_data(data_path)

    network = autocrit.nn.networks.Network.from_json(
        data, network_json_path)

    if network.batch_size is not None:
        loss = network.loss_on_random_batch
    else:
        loss = network.loss

    if experiment_type == "optimization":
        experiment = autocrit.experiments.OptimizationExperiment.from_json(
            loss, experiment_json_path)
    elif experiment_type == "critfinder":
        experiment = autocrit.experiments.CritFinderExperiment.from_json(
            loss, experiment_json_path)
    else:
        raise NotImplementedError("experiment_type {} not understood"
                                  .format(experiment_type))

    return data, network, experiment


def unpickle(pfn):
    with open(pfn, 'rb') as pf:
        dct = pickle.load(pf, encoding='bytes')
    return dct
