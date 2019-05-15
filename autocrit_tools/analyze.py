import autograd.numpy as np


def plot_trajectories(cp_df, key, ax, plot_func="plot", func=lambda x: x,
                      subplots_kwargs=None, plot_func_kwargs=None):

    if plot_func_kwargs is None:
        plot_func_kwargs = {}

    for ii, row in cp_df.iterrows():
        ys = func(row[key])
        if plot_func == "plot":
            ax.plot(ys, **plot_func_kwargs)

    return ax


def compute_maps(weight_lists):
    maps = []
    for weight_list in weight_lists:
        # could be more generically implemented as rfold with dot
        maps.append(np.dot(weight_list[1], weight_list[0]))
    return maps


def extract_weight_lists(cp_df, network, index=-1):
    weight_lists = []
    for _, row in cp_df.iterrows():
        weight_lists.append(network.extract_weights(row["thetas"][index]))
    return weight_lists
