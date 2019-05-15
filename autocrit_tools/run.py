"""Utilities for interacting with the scripts programmatically in Python.
"""
import os
import subprocess


def run_from_args_list(script, args_list):
    return subprocess.run(["python", script] + args_list,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def run_script_pair(setup_args, run_args, dir, pair="optimization"):

    setup_script = os.path.join(dir, "setup_{}_experiment.py".format(pair))
    run_script = os.path.join(dir, "run_{}_experiment.py".format(pair))
    setup_cp = run_from_args_list(setup_script, setup_args)

    if setup_cp.returncode > 0:
        return setup_cp, None

    run_cp = run_from_args_list(run_script, run_args)

    return setup_cp, run_cp


def make_optimizer_setup_args(data_dir, network_ID, **kwargs):
    args = ["--data_dir", data_dir,
            "--network_ID", network_ID]

    args = add_kwargs(kwargs, args)

    return args


def make_optimizer_run_args(optimizer_dir, num_steps, **kwargs):
    args = ["--optimizer_dir", optimizer_dir]

    args = add_kwargs(kwargs, args)

    args += [str(num_steps)]

    return args


def make_critfinder_setup_args(optimizer_dir, finder, **kwargs):
    args = []

    args = add_kwargs(kwargs, args)

    args += [optimizer_dir, finder]

    return args


def make_critfinder_run_args(critfinder_dir, output_ID, num_iters, **kwargs):
    args = []

    args = add_kwargs(kwargs, args)

    args += [critfinder_dir, output_ID, str(num_iters)]

    return args


def add_kwargs(kwargs, args):
    for arg_name, arg_value in kwargs.items():
        if arg_value is not None:
            args += ["--" + arg_name]
            if str(arg_value).strip() != "":
                args += [str(arg_value)]
    return args


def show_cp(cp):
    print("args:\n", cp.args)
    print("stdout:\n", cp.stdout.decode("utf-8"))
    print("stderr:\n", cp.stderr.decode("utf-8"))


def print_results(results):
    for ii, result in enumerate(results):
        print("=" * 10 + "\n" + str(ii) + ":")

        if result[0].returncode > 0:
            print("error  in setup:")
            show_cp(result[0])

        elif result[1].returncode > 0:
            print("error in run:")
            show_cp(result[1])
