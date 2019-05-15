"""Provides a class, ExperimentPaths, that manages a collection of pathlib.Paths
that point to the various components of a critical point-finding experiment
using autocrit. Primarily designed for use with the code in scripts/.

See the docs for the ExperimentPaths class for details.
"""
import pathlib


class ExperimentPaths(object):
    """Class to encapsulate all relative path logic for critical point-finding experiments
    and store paths as pathlib.Path objects.

    Also creates all inferrable directories on construction.

    """
    datafilename = "data.npz"

    def __init__(self, data_ID, root=".", network_ID=None, optimizer_ID=None, critfinder_ID=None):
        """
        The complete file structure looks like this:

            root/
            ├── data_ID
            │   ├── data.npz
            │   └── network_ID
            │       ├── network.json
            │       └── optimizer_ID
            │           ├── optimizer.json
            │           ├── critfinder_ID
            │           │   ├── experiment.json
            │           │   ├── finder.json
            │           │   └── outputs
            │           │       └── 0000.npz
            │           └── trajectories
            │               └── 0000.npz
        """
        self.data_ID = data_ID
        self.data_dir = to_path(root) / self.data_ID
        self.data = self.data_dir / ExperimentPaths.datafilename

        self.set_network_ps(network_ID)
        self.set_optimizer_ps(optimizer_ID)
        self.set_critfinder_ps(critfinder_ID)

        self.directories = {"data": self.data_dir,
                            "network": self.network_dir,
                            "optimizer": self.optimizer_dir,
                            "optimizer_traj": self.optimizer_traj_dir,
                            "finder": self.finder_dir,
                            "finder_out": self.finder_out_dir}

        self.jsons = {"network": self.network,
                      "optimizer": self.optimizer,
                      "experiment": self.experiment,
                      "finder": self.finder}
        self.make()

    def make(self):
        for dir in self.directories.values():
            if dir is not None:
                dir.mkdir(parents=True, exist_ok=True)

    def set_network_ps(self, network_ID):
        self.network_ID = network_ID
        if self.network_ID is not None:
            self.network_dir = self.data_dir / network_ID
            self.network = self.network_dir / "network.json"
        else:
            self.network_dir = self.network = None

    def set_optimizer_ps(self, optimizer_ID):
        self.optimizer_ID = optimizer_ID
        if self.optimizer_ID is not None:
            self.optimizer_dir = self.network_dir / optimizer_ID
            self.optimizer = self.optimizer_dir / "optimizer.json"
            self.optimizer_traj_dir = self.optimizer_dir / "trajectories"
        else:
            self.optimizer_dir = self.optimizer = self.optimizer_traj_dir = None

    def set_critfinder_ps(self, critfinder_ID):
        self.critfinder_ID = critfinder_ID
        if self.critfinder_ID is not None:
            assert self.optimizer_dir is not None
            self.finder_dir = self.optimizer_dir / critfinder_ID
            self.finder = self.finder_dir / "finder.json"
            self.experiment = self.finder_dir / "experiment.json"
            self.finder_out_dir = self.finder_dir / "outputs"
        else:
            self.finder_dir = self.experiment_dir = self.finder_out_dir = None
            self.finder = self.experiment = None

    @classmethod
    def from_critfinder_dir(cls, critfinder_dir):
        critfinder_dir = to_path(critfinder_dir)

        critfinder_ID = critfinder_dir.stem
        optimizer_dir = critfinder_dir.parent

        return cls.from_optimizer_dir(optimizer_dir,
                                      critfinder_ID=critfinder_ID)

    @classmethod
    def from_optimizer_dir(cls, optimizer_dir, critfinder_ID=None):
        optimizer_dir = to_path(optimizer_dir)

        optimizer_ID = optimizer_dir.stem
        network_dir = optimizer_dir.parent

        return cls.from_network_dir(network_dir,
                                    optimizer_ID=optimizer_ID, critfinder_ID=critfinder_ID)

    @classmethod
    def from_network_dir(cls, network_dir, optimizer_ID=None, critfinder_ID=None):
        network_dir = to_path(network_dir)

        data_dir = network_dir.parent
        network_ID = network_dir.stem

        return cls.from_data_dir(data_dir,
                                 network_ID=network_ID, optimizer_ID=optimizer_ID,
                                 critfinder_ID=critfinder_ID)

    @classmethod
    def from_data_dir(cls, data_dir, network_ID=None, optimizer_ID=None, critfinder_ID=None):
        data_dir = to_path(data_dir)

        root_dir = data_dir.parent
        data_ID = data_dir.stem

        return cls(data_ID, root=root_dir, network_ID=network_ID,
                   optimizer_ID=optimizer_ID, critfinder_ID=critfinder_ID)

    def reroot(self, new_root):
        return ExperimentPaths(self.data_ID, root=new_root, network_ID=self.network_ID,
                               optimizer_ID=self.optimizer_ID, critfinder_ID=self.critfinder_ID)


def to_path(path):
    if not isinstance(path, pathlib.Path):
        return pathlib.Path(path)
    return path
