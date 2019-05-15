import argparse
import sys

import autograd.numpy as np

from autocrit_tools.path import ExperimentPaths
import autocrit_tools.load


def main(args):
    paths = ExperimentPaths.from_critfinder_dir(args.critfinder_dir)
    experiment_dict = autocrit_tools.load.open_json(paths.experiment)

    trajectory = fetch_trajectory(experiment_dict["trajectory_path"])

    output_ID = args.output_ID
    output_filename = output_ID + ".npz"
    output_path = paths.finder_out_dir / output_filename

    _, _, finder_experiment = autocrit_tools.load.from_paths(
        paths.data, paths.network, paths.finder,
        experiment_type="critfinder")

    init_theta = initialize_theta(
        finder_experiment, experiment_dict["init_theta"], trajectory,
        experiment_dict["theta_perturb"])

    finder_experiment.run(init_theta, args.num_iters)

    finder_experiment.save_results(output_path)


def initialize_theta(experiment, init_theta_str, thetas, theta_perturb):
    if init_theta_str == "uniform":
        init_theta = experiment.uniform(thetas)
    elif init_theta_str == "uniform_f":
        init_theta = experiment.uniform_f(thetas)
    elif init_theta_str.endswith(".npy"):
        init_theta = np.load(init_theta_str)
    else:
        raise NotImplementedError("init_theta_str {} not understood"
                                  .format(init_theta_str))

    if theta_perturb is not None:
        perturb_stdev = np.sqrt(10 ** theta_perturb)
        init_theta += perturb_stdev * np.random.standard_normal(size=init_theta.shape)

    return init_theta


def fetch_trajectory(trajectory_path):
    if trajectory_path.endswith(".npz"):
        results_npz = np.load(trajectory_path)
        trajectory = results_npz["theta"]
    else:
        raise NotImplementedError("trajectory_path {} not understood"
                                  .format(trajectory_path))
    return trajectory


def setup_parser():
    parser = argparse.ArgumentParser(
        description="Load and run a critfinder experiment from its constituent files.")

    parser.add_argument("critfinder_dir", type=str)
    parser.add_argument("output_ID", type=str)
    parser.add_argument("num_iters", type=int)

    return parser


if __name__ == "__main__":
    parser = setup_parser()
    main(parser.parse_args())
    sys.exit(0)
