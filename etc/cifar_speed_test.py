import argparse
import os
import sys
import time
import warnings

import autograd.numpy as np

import autocrit.nn as nn
import autocrit.utils.math

warnings.simplefilter("ignore")
CIFAR10_DIR = os.path.join("..", "data", "cifar10")


def main(batch_sizes, step_cts, on_full):
    raws, labels = load_cifar(CIFAR10_DIR)
    images, one_hots = preprocess(raws, labels)
    cifar10 = (images, one_hots)

    init_theta = None

    print_header()

    for batch_size in batch_sizes:
        init_loss = None

        gap_conv = define_gap_conv(cifar10, batch_size=batch_size, num_labels=10)

        if init_theta is None:
            init_theta = (1 / np.sqrt(160)) * gap_conv.initialize()

        if init_loss is None:
            if on_full:
                init_loss = gap_conv.loss(init_theta)
            else:
                init_loss = gap_conv.loss_on_random_batch(init_theta)

        for step_ct in step_cts:
            momentum_opt = autocrit.optimizers.MomentumOptimizer(
                gap_conv.loss_on_random_batch, momentum=0.9, lr=0.05)

            t_start = time.time()
            final_theta = momentum_opt.run(init_theta, step_ct)
            t_finish = time.time()
            elapsed_t = t_finish - t_start

            if on_full:
                final_loss = gap_conv.loss(final_theta)
            else:
                final_loss = gap_conv.loss_on_random_batch(final_theta)

            acc = compute_accuracy(gap_conv, final_theta, on_full)

            print_result(batch_size, step_ct, elapsed_t, init_loss, final_loss, acc)


def define_gap_conv(cifar10, batch_size, num_labels):
    gap_conv = nn.networks.Network(cifar10,
                                   [nn.layers.ConvLayer((3, 3), 16),
                                    nn.layers.PointwiseNonlinearLayer("relu"),
                                    nn.layers.MaxPoolLayer((2, 2)),
                                    nn.layers.ConvLayer((4, 4), num_labels),
                                    nn.layers.GlobalAvgPoolLayer(),
                                    nn.layers.SqueezeLayer()
                                    ],
                                   cost_str="softmax_cross_entropy",
                                   batch_size=batch_size)
    return gap_conv


def compute_accuracy(network, theta, full_acc):
    if full_acc:
        outs = network.forward_pass(network.data.x, theta)
        true_labels = np.argmax(network.data.y, axis=0)
    else:
        dataset_size = network.data.x.shape[-1]
        idxs = np.random.choice(dataset_size, size=network.batch_size)
        outs = network.forward_pass(network.data.x[..., idxs], theta)
        true_labels = np.argmax(network.data.y[..., idxs], axis=0)

    network_labels = np.argmax(outs, axis=0)

    return np.mean(true_labels == network_labels)


def print_result(*args):
    print("\t".join([str(arg) for arg in args]))


def print_header():
    columns = ["batch_size", "step_ct", "elapsed_t", "init_loss", "final_loss", "acc"]

    print("\t".join(columns))


def load_cifar(cifar10_dir):

    cifar10_batches_dir = os.path.join(cifar10_dir, "cifar-10-batches-py")

    test_batch_filename = os.path.join(cifar10_batches_dir, "data_batch_1")

    test_batch = autocrit.utils.load.unpickle(test_batch_filename)

    labels = test_batch[b"labels"]
    raws = test_batch[b"data"]

    return raws, labels


def preprocess(raw_images, labels):

    images = cifar10_raws_to_images(raw_images)
    images = autocrit.utils.math.rescale(images)

    num_labels = max(labels) + 1
    one_hot_mat = np.eye(num_labels)
    one_hots = np.asarray([one_hot_mat[label] for label in labels]).T

    return images, one_hots


def cifar10_raw_to_image(raw):
    CIFAR10_SIDE_LENGTH = 32
    image = np.reshape(raw, (CIFAR10_SIDE_LENGTH, CIFAR10_SIDE_LENGTH, 3), order="F")
    image = np.moveaxis(image, [0, 1], [1, 0])
    return image


def cifar10_raws_to_images(raws):
    images = np.asarray([cifar10_raw_to_image(raw) for raw in raws])
    images = np.moveaxis(images, [0], [-1])
    return images


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_sizes', metavar='batch_size', type=int, nargs='+',
                        help='batch sizes to run')
    parser.add_argument('--step_cts', metavar='step_ct', type=int, nargs='+',
                        help='number of steps to run')
    parser.add_argument('--on_full', action='store_true',
                        help='supply to calculate loss/accuracy on full dataset')

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    sys.exit(main(args.batch_sizes, args.step_cts, args.on_full))
