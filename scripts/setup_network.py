import argparse
import os
import sys

import autograd.numpy as np

import autocrit

import autocrit_tools.load as load
from autocrit_tools.path import ExperimentPaths
from autocrit_tools.util import random_string

DEFAULT_VERBOSITY = 0

DEFAULT_RESULTS_PATH = os.path.join("results")

DEFAULT_TASK = "autoencoding"

DEFAULT_K = 16
DEFAULT_N = 10000

DEFAULT_P = 4
DEFAULT_REGULARIZER = "l2"
DEFAULT_REGULARIZATION_PARAMETER = 0.0
DEFAULT_NONLINEARITY = "none"


def main(args):

    network_ID = args.ID or random_string(6)
    if args.verbosity > 0:
        print("creating files for {}".format(network_ID))

    layers_path = args.layers_path
    layers = load_layers(layers_path)

    paths = ExperimentPaths(args.data_ID, args.results_path, network_ID)
    data = handle_data(args, paths.data)

    if args.task in ["autoencoding", "regression"]:
        cost_str = "mean_squared_error"
    else:
        assert args.task == "classification"
        cost_str = "softmax_cross_entropy"

    MLP = autocrit.nn.networks.Network(
        data, layer_specs=layers,
        regularizer_str=args.regularizer, regularization_parameter=args.regularization_parameter,
        cost_str=cost_str, batch_size=args.batch_size)

    MLP.to_json(paths.network)


def load_layers(layers_path):
    return load.open_json(layers_path)


def handle_data(args, data_path):
    if args.generate_data:
        xs = generate_data(args.k, args.N, args.zero_centering)
        np.savez(data_path, xs=xs)

    data = load.fetch_data(data_path)

    return data


def generate_data(k, N, zero_centering):
    Sigma = np.diag(range(1, k + 1))
    xs = np.random.multivariate_normal(mean=np.zeros(k), cov=Sigma, size=N).T

    if zero_centering == "subtract_mean":
        xs = xs - np.mean(xs, axis=1)[:, None]
    elif zero_centering == "add_point":
        xs = xs[:, :-1]
        xs = xs - np.mean(xs, axis=1)[:, None]
        approx_mu = np.mean(xs, axis=1)[:, None]
        xs = np.hstack([xs, -(N - 1) * approx_mu])

    return xs


def setup_parser():
    parser = argparse.ArgumentParser(
        description="Create necessary files for an optimization experiment.")

    # PROGRAM
    parser.add_argument("-v", dest="verbosity",
                        action="store_const", const=1, default=0,
                        help="verbosity flag")

    # PATHS
    parser.add_argument("--results_path", type=str, default=DEFAULT_RESULTS_PATH,
                        help="top-level directory for results. " +
                        "default is {}".format(DEFAULT_RESULTS_PATH))
    parser.add_argument("--ID", type=str, default=None,
                        help="identifier for this network." +
                        "provide to over-ride default behavior, generating a random ID.")

    # DATA
    parser.add_argument("--data_ID", type=str,
                        help="ID to name directory where data is stored (as data.npz)")
    parser.add_argument("--task", type=str, default=DEFAULT_TASK,
                        choices=["autoencoding", "classification", "regression"],
                        help="task to apply to data. default is {}".format(DEFAULT_TASK))
    parser.add_argument("--N", type=int, default=DEFAULT_N,
                        help="number of data points to generate, if generate_data flag active. " +
                        "default is {}".format(DEFAULT_N))
    parser.add_argument("--k", type=int, default=DEFAULT_K,
                        help="number of input dimensions, " +
                        "if generate_data flag active. " +
                        "default is {}".format(DEFAULT_K))
    parser.add_argument("--zero_centering", type=str, default="none",
                        choices=["none", "subtract_mean", "add_point"],
                        help="type of zero centering to apply to the data: " +
                        "subtract_mean, subtract the mean; " +
                        "add_point, attempt to add a number to the sample " +
                        "such that the data has mean exactly zero on each dimension; " +
                        "none, meaning do nothing. " +
                        "add_point is less subject to floating point error. " +
                        "Default is none.")
    parser.add_argument("--generate_data", action="store_const", const=True, default=False,
                        help="flag to generate gaussian data for an autoencoding task.")

    # NETWORK
    parser.add_argument("--layers_path",
                        help="path to json file specifying the network's layers.")
    parser.add_argument("--regularizer", type=str, default=DEFAULT_REGULARIZER,
                        choices=["none", "l1", "l2"],
                        help="type of regularization term to apply to weights. " +
                        "default is {}, but see --regularization_parameter"
                        .format(DEFAULT_REGULARIZER))
    parser.add_argument("--regularization_parameter", type=float,
                        default=DEFAULT_REGULARIZATION_PARAMETER,
                        help="multiplicative factor to apply to regularization cost. " +
                        "default is {}".format(DEFAULT_REGULARIZATION_PARAMETER))
    parser.add_argument("--batch_size", type=int, default=None,
                        help="size of batches to draw during optimization. " +
                        "Defaults to full batch")

    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    main(args)
    sys.exit(0)
