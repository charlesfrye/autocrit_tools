import argparse
import sys

import autograd.numpy as np

from autocrit_tools.path import ExperimentPaths
import autocrit_tools.load


def main(num_steps, paths, trajectory_ID, init_theta=None):

    _, network, optimization_experiment = autocrit_tools.load.from_paths(
        paths.data, paths.network, paths.optimizer)

    init_theta = initialize_theta(init_theta, network)

    optimization_experiment.run(init_theta, num_steps)

    optimization_experiment.save_results(paths.optimizer_traj_dir / (trajectory_ID + ".npz"))


def setup_paths(optimizer_dir):
    return ExperimentPaths.from_optimizer_dir(optimizer_dir)


def initialize_theta(init_theta, network):
    if init_theta is None:
        init_theta = network.initialize()
    elif isinstance(init_theta, str):
        init_theta = np.load(init_theta)

    return init_theta


def construct_parser():

    parser = argparse.ArgumentParser(
        description='Execute an optimization experiment and save the resulting trajectory.')

    parser.add_argument("num_steps", type=int,
                        help="Number of steps of optimizer to apply.")

    parser.add_argument("--optimizer_dir", type=str, default=None,
                        help="path to directory containing optimizer.json and " +
                        "contained within a directory containing network.json, " +
                        "which should be contained within a directory containing data.npz"
                        )

    parser.add_argument("--trajectory_ID", type=str, default=str(0).zfill(4),
                        help="identifying string for trajectory. used as filename.")

    parser.add_argument("--init_theta", type=str, default=None,
                        help="path to initial parameter values. " +
                        "If not provided, uses network.initialize method to generate values.")

    return parser


if __name__ == "__main__":
    parser = construct_parser()
    args = parser.parse_args()
    paths = setup_paths(args.optimizer_dir)
    main(args.num_steps, paths, args.trajectory_ID, args.init_theta)
    sys.exit(0)
