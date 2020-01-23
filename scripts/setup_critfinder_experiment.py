import argparse
import json
import sys

import autograd.numpy as np

import autocrit.experiments

from autocrit_tools.path import ExperimentPaths
from autocrit_tools.util import random_string


DEFAULT_VERBOSITY = 0

DEFAULT_INIT_THETA = "uniform"
DEFAULT_LOG_KWARGS = {"track_theta": True,
                      "track_g": True,
                      "track_f": True}

DEFAULT_ALPHA = autocrit.defaults.DEFAULT_ALPHA
DEFAULT_BETA = autocrit.defaults.DEFAULT_BETA
DEFAULT_RHO = autocrit.defaults.DEFAULT_RHO
DEFAULT_RHO_PURE = autocrit.defaults.DEFAULT_RHO_PURE
DEFAULT_GAMMA = autocrit.defaults.DEFAULT_GAMMA

DEFAULT_RTOL = autocrit.defaults.DEFAULT_RTOL
DEFAULT_MAXIT = autocrit.defaults.DEFAULT_MAXIT

DEFAULT_NEWTON_STEP_SIZE = DEFAULT_ALPHA

# TODO: reconcile gamma settings across DEFAULTS
DEFAULT_TR_GAMMA_MX, DEFAULT_TR_GAMMA_K = -1, 6

DEFAULT_GNM_MINIMIZER = "btls"
DEFAULT_GNM_MINIMIZER_PARAMS = {"alpha": DEFAULT_ALPHA,
                                "beta": DEFAULT_BETA}
DEFAULT_GNM_CRITERION = "wolfe"
DEFAULT_GNM_CRITERION_PARAMS = {"rho": DEFAULT_RHO,
                                "gamma": DEFAULT_GAMMA}

DEFAULT_LR = DEFAULT_ALPHA
# TODO: check that momentum is defined the same across implementations
DEFAULT_MOMENTUM = autocrit.defaults.DEFAULT_MOMENTUM


def main(args):

    ID = args.ID or random_string(6)
    paths = ExperimentPaths.from_optimizer_dir(args.optimizer_dir, critfinder_ID=ID)

    finder_str, finder_kwargs = extract_finder_info(args)
    finder_experiment = autocrit.CritFinderExperiment(lambda x: None, finder_str, finder_kwargs)
    finder_experiment.to_json(paths.finder)

    experiment_dict = make_experiment_dict(args, paths, ID)
    write_experiment_json(experiment_dict, paths.experiment)


def make_experiment_dict(args, paths, ID):

    optimizer_path = paths.optimizer

    trajectory_path = paths.optimizer_traj_dir / (args.trajectory_ID + ".npz")

    experiment_dict = {"optimizer_path": str(optimizer_path),
                       "ID": ID,
                       "trajectory_path": str(trajectory_path),
                       "init_theta": args.init_theta,
                       "theta_perturb": args.theta_perturb,
                       "finder_json": str(paths.finder)
                       }

    return experiment_dict


def write_experiment_json(experiment_dict, experiment_json_path):
    with open(experiment_json_path, "w") as fp:
        json.dump(experiment_dict, fp)


def extract_finder_info(args):
    finder_str = args.finder
    finder_kwargs = {"log_kwargs": DEFAULT_LOG_KWARGS}

    if finder_str.startswith("newton"):
        finder_kwargs.update(extract_newton_kwargs(args))
        if finder_str.endswith("MR"):
            finder_kwargs.update(extract_mr_kwargs(args))
        elif finder_str.endswith("TR"):
            finder_kwargs.update(extract_tr_kwargs(args))
        else:
            raise NotImplementedError("finder_str {0} not understood".format(finder_str))
    elif finder_str == "gnm":
        finder_kwargs.update(extract_gnm_kwargs(args))
    else:
        raise NotImplementedError("finder_str {0} not understood".format(finder_str))
    return finder_str, finder_kwargs


def extract_newton_kwargs(args):
    newton_kwargs = {}
    return newton_kwargs


def extract_tr_kwargs(args):
    tr_kwargs = {"step_size": args.newton_step_size,
                 "gammas": construct_gammas(args)}
    tr_kwargs.update(extract_minresqlp_kwargs(args))
    return tr_kwargs


def construct_gammas(args):
    mx, k = args.gamma_mx, args.gamma_k
    gammas = np.logspace(num=k, start=mx, stop=mx - k, endpoint=False).tolist()
    return gammas


def extract_mr_kwargs(args):
    mr_kwargs = {"alpha": args.alpha,
                 "beta": args.beta,
                 "rho": args.rho,
                 "check_pure": args.check_pure,
                 "rho_pure": args.rho_pure}
    mr_kwargs.update(extract_minresqlp_kwargs(args))
    return mr_kwargs


def extract_minresqlp_kwargs(args):
    return {"rtol": args.rtol, "maxit": args.maxit}


def extract_gnm_kwargs(args):
    gnm_kwargs = {"minimizer_str": args.minimizer}
    gnm_kwargs["minimizer_params"] = extract_minimizer_params(args)

    return gnm_kwargs


def extract_minimizer_params(args):
    minimizer_params = {}
    if args.minimizer in ["gd", "momentum"]:
        minimizer_params["lr"] = args.lr
        if args.minimizer == "momentum":
            minimizer_params["momentum"] = args.momentum
    elif args.minimizer == "btls":
        minimizer_params["alpha"] = args.alpha
        minimizer_params["beta"] = args.beta
        minimizer_params["rho"] = args.rho
        minimizer_params.update(extract_criterion_info(args))
    else:
        raise NotImplementedError("minimizer_str {0} not understood"
                                  .format(args.minimizer))

    return minimizer_params


def extract_criterion_info(args):

    if args.criterion in ["wolfe", "roosta"]:
        criterion_info = {"criterion": args.criterion}
        if args.criterion == "wolfe":
            criterion_info["gamma"] = args.gamma
    else:
        raise NotImplementedError("criterion_str {0} not understood"
                                  .format(args.criterion))

    return criterion_info


def setup_parser():
    parser = argparse.ArgumentParser(
        description="Creates the necessary files to run a critfinder experiment.")

    # PROGRAM
    parser.add_argument("-v",
                        action="store_const", dest="verbosity", const=1, default=DEFAULT_VERBOSITY,
                        help="verbosity flag")

    # EXPERIMENT
    parser.add_argument("optimizer_dir", type=str,
                        help="directory of optimization.")
    parser.add_argument("--trajectory_ID", type=str, default="0000",
                        help="identifier string for trajectory. used as filename prefix to "
                        ".npz suffix. default is 0000.")
    parser.add_argument("--init_theta", type=str, default=DEFAULT_INIT_THETA,
                        help="either a string identifying an initialization strategy " +
                        "based on a trajectory or a string path to an init_theta npy file. " +
                        "initialization strategies are {uniform, uniform_f} for selecting " +
                        "points uniformly from the trajectory or " +
                        "with a uniform distribution on heights.")
    parser.add_argument("--theta_perturb", type=float, default=None,
                        help="Logarithm base 10 of amount to perturb theta" +
                        "initialized from optimization trajectory." +
                        "sets variance of additive gaussian noise." +
                        "default is None, which results in no perturbation.")
    parser.add_argument("--ID", type=str, default=None,
                        help="if not provided, a random string is generated as ID.")

    # CRITFINDER
    parser.add_argument("finder", type=str, choices=["gnm", "newtonMR", "newtonTR"],
                        help="string identifying the finder to use.")

    # btls
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                        help="used by all BTLS. " +
                        "initial value for step size. " +
                        "default is {}".format(DEFAULT_ALPHA))
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA,
                        help="used by all BTLS. " +
                        "multiplicative factor by which to decrease step size on " +
                        "each failed line search step. " +
                        "default is {}".format(DEFAULT_BETA))
    parser.add_argument("--rho", type=float, default=DEFAULT_RHO,
                        help="used in armijo check in all BTLS. " +
                        "hyperparameter for strictness of sufficient decrease condition. " +
                        "default is {}".format(DEFAULT_RHO))
    parser.add_argument("--check_pure",
                        action="store_const", dest="check_pure",
                        const=True, default=False,
                        help="used in armijo check in Newton MR. " +
                        "if True, always check whether a 'pure' Newton step " +
                        "of step size 1 is acceptable. see --rho_pure")
    parser.add_argument("--rho_pure", type=float, default=DEFAULT_RHO_PURE,
                        help="used in armijo check in Newton MR. " +
                        "hyperparameter for strictness of sufficient decrease condition " +
                        "for steps of size 1. only used if --check_pure provided." +
                        "default is {}".format(DEFAULT_RHO_PURE))

    # gnm
    parser.add_argument("--minimizer", type=str, default=DEFAULT_GNM_MINIMIZER,
                        help="minimizer to use on gnm. " +
                        "default is {}".format(DEFAULT_GNM_MINIMIZER))
    parser.add_argument("--criterion", type=str, default=DEFAULT_GNM_CRITERION,
                        help="stopping criterion to use on btls for gnm. " +
                        "default is {}".format(DEFAULT_GNM_CRITERION))
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA,
                        help="used in wolfe criterion on gnm. " +
                        "hyperparameter for strictness of curvature condition. " +
                        "default is {}".format(DEFAULT_GAMMA))

    # gd and momentum
    parser.add_argument("--lr", type=float, default=DEFAULT_LR,
                        help="used by gnm. " +
                        "learning rate for gradient descent or momentum minimizer. " +
                        "default is {}".format(DEFAULT_LR))
    parser.add_argument("--momentum", type=float, default=DEFAULT_MOMENTUM,
                        help="used by gnm. " +
                        "fraction of momentum term to preserve across steps. "
                        "default is {}".format(DEFAULT_MOMENTUM))

    # minresqlp
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    parser.add_argument("--maxit", type=int, default=DEFAULT_MAXIT)

    # newton-tr
    parser.add_argument("--newton_step_size", type=float, default=DEFAULT_NEWTON_STEP_SIZE,
                        help="used by non-BTLS Newtons. " +
                        "step size for Newton method. " +
                        "default is {}".format(DEFAULT_NEWTON_STEP_SIZE))
    parser.add_argument("--gamma_mx", type=float, default=DEFAULT_TR_GAMMA_MX,
                        help="used by trust-region Newton. " +
                        "maximum order of magnitude for trust region size parameter. " +
                        "default is {}".format(DEFAULT_TR_GAMMA_MX))
    parser.add_argument("--gamma_k", type=int, default=DEFAULT_TR_GAMMA_K,
                        help="used by trust-region Newton. " +
                        "number of orders of magnitude over which to vary " +
                        "trust region size parameter. " +
                        "default is {}".format(DEFAULT_TR_GAMMA_K))

    return parser


if __name__ == "__main__":
    parser = setup_parser()
    main(parser.parse_args())
    sys.exit(0)
