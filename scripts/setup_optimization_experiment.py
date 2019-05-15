import argparse
import sys

import autocrit

import autocrit_tools.load as load
from autocrit_tools.path import ExperimentPaths
from autocrit_tools.util import random_string

DEFAULT_VERBOSITY = 0

DEFAULT_OPTIMIZER = "gd"
DEFAULT_OPTIMIZER_LR = 0.1
DEFAULT_OPTIMIZER_MOMENTUM = 0.1

DEFAULT_LOG_KWARGS = {"track_theta": True, "track_f": True, "track_g": False}


def main(args):

    ID = args.ID or random_string(6)
    if args.verbosity > 0:
        print("creating files for {}".format(ID))

    paths = ExperimentPaths.from_data_dir(args.data_dir,
                                          network_ID=args.network_ID, optimizer_ID=ID)
    data = load.fetch_data(paths.data)

    optimizer_kwargs = extract_optimizer_kwargs(args)
    log_kwargs = setup_log_kwargs(args)

    MLP = autocrit.nn.networks.Network.from_json(
        data, paths.network)

    if MLP.batch_size is not None:
        loss = MLP.loss_on_random_batch
    else:
        loss = MLP.loss

    optimization_experiment = autocrit.OptimizationExperiment(
        loss,
        optimizer_str=args.optimizer,
        optimizer_kwargs=optimizer_kwargs,
        log_kwargs=log_kwargs,
        seed=args.seed)

    optimization_experiment.to_json(paths.optimizer)


def extract_optimizer_kwargs(args):
    optimizer_kwargs = {"lr": args.optimizer_lr}
    if args.optimizer == "momentum":
        if args.optimizer_momentum is not None:
            optimizer_kwargs["momentum"] = args.optimizer_momentum
        else:
            optimizer_kwargs["momentum"] = DEFAULT_OPTIMIZER_MOMENTUM

    return optimizer_kwargs


def setup_log_kwargs(args):
    log_kwargs = DEFAULT_LOG_KWARGS.copy()
    if args.log_gradients:
        log_kwargs["g_theta"] = True

    return log_kwargs


def setup_parser():
    parser = argparse.ArgumentParser(
        description="Create necessary files for an optimization experiment.")

    # PROGRAM
    parser.add_argument("-v", dest="verbosity",
                        action="store_const", const=1, default=0,
                        help="verbosity flag")

    # PATHS
    parser.add_argument("--ID", type=str, default=None,
                        help="identifier for this optimization problem." +
                        "provide to over-ride default behavior, generating a random ID.")
    parser.add_argument("--data_dir", type=str,
                        help="path to directory containing dataset.")

    # NETWORK
    parser.add_argument("--network_ID", type=str,
                        help="identifier for network. must be located inside data_dir " +
                        "as subdirectory and contain network.json.")

    # OPTIMIZER
    parser.add_argument("--optimizer", type=str, default=DEFAULT_OPTIMIZER,
                        help="optimizer to apply to network loss. " +
                        "default is {}".format(DEFAULT_OPTIMIZER),
                        choices=["gd", "momentum"])
    parser.add_argument("--optimizer_lr", type=float, default=DEFAULT_OPTIMIZER_LR,
                        help="learning rate for optimizer. " +
                        "default is {}".format(DEFAULT_OPTIMIZER_LR))
    parser.add_argument("--optimizer_momentum", type=float, default=None,
                        help="momentum level for momentum optimizer. " +
                        "default is {}".format(DEFAULT_OPTIMIZER_MOMENTUM))
    parser.add_argument("--seed", type=int, default=None,
                        help="seed value for np.random and random. if not provided, " +
                        "defaults to a shared global value in autocrit.experiments.")
    parser.add_argument("--log_gradients",
                        dest="log_gradients", action="store_const", const=True, default=False,
                        help="flag to log gradients of loss with respect to theta" +
                        "during training.")

    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    main(args)
    sys.exit(0)
