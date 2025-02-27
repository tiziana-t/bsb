import abc, random, types
import numpy as np
from ..helpers import ConfigurableClass
from ..reporting import report
from ..exceptions import *
from time import time
import itertools


class SimulatorAdapter(ConfigurableClass):
    def __init__(self):
        super().__init__()
        self.cell_models = {}
        self.connection_models = {}
        self.devices = {}
        self.entities = {}
        self._progress_listeners = []

    def get_configuration_classes(self):
        if not hasattr(self.__class__, "simulator_name"):
            raise AttributeMissingError(
                "The SimulatorAdapter {} is missing the class attribute 'simulator_name'".format(
                    self.__class__
                )
            )
        # Check for the 'configuration_classes' class attribute
        if not hasattr(self.__class__, "configuration_classes"):
            raise AdapterError(
                "The '{}' adapter class needs to set the 'configuration_classes' class attribute to a dictionary of configurable classes (str or class).".format(
                    self.simulator_name
                )
            )
        classes = self.configuration_classes
        keys = ["cell_models", "connection_models", "devices"]
        # Check for the presence of required classes
        for requirement in keys:
            if requirement not in classes:
                raise AdapterError(
                    "{} adapter: The 'configuration_classes' dictionary requires a class under the '{}' key.".format(
                        self.simulator_name, requirement
                    )
                )
        # Test if they are all children of the ConfigurableClass class
        for class_key in keys:
            if not issubclass(classes[class_key], ConfigurableClass):
                raise AdapterError(
                    "{} adapter: The configuration class '{}' should inherit from ConfigurableClass".format(
                        self.simulator_name, class_key
                    )
                )
        return self.configuration_classes

    @abc.abstractmethod
    def prepare(self, hdf5, simulation_config):
        """
        This method turns a stored HDF5 network architecture and returns a runnable simulator.

        :returns: A simulator prepared to run a simulation according to the given configuration.
        """
        pass

    @abc.abstractmethod
    def simulate(self, simulator):
        """
        Start a simulation given a simulator object.
        """
        pass

    @abc.abstractmethod
    def collect_output(self, simulator):
        """
        Collect the output of a simulation that completed
        """
        pass

    @abc.abstractmethod
    def get_rank(self):
        """
        Return the rank of the current node.
        """
        pass

    @abc.abstractmethod
    def get_size(self):
        """
        Return the size of the collection of all distributed nodes.
        """
        pass

    @abc.abstractmethod
    def broadcast(self, data, root=0):
        """
        Broadcast data over MPI
        """
        pass

    def start_progress(self, duration):
        """
        Start a progress meter.
        """
        self._progdur = duration
        self._progstart = self._last_progtic = time()
        self._progtics = 0

    def progress(self, step):
        """
        Report simulation progress.
        """
        now = time()
        tic = now - self._last_progtic
        self._progtics += 1
        el = now - self._progstart
        report(
            f"Simulated {step}/{self._progdur}ms.",
            f"{el:.2f}s elapsed.",
            f"Simulated tick in {tic:.2f}.",
            f"Avg tick {el / self._progtics:.4f}s",
            level=3,
            ongoing=False,
        )
        progress = types.SimpleNamespace(
            progression=step, duration=self._progdur, time=time()
        )
        for listener in self._progress_listeners:
            listener(progress)
        self._last_progtic = now
        return progress

    def step_progress(self, duration, step=1):
        steps = itertools.chain(np.arange(0, duration), (duration,))
        a, b = itertools.tee(steps)
        next(b, None)
        yield from zip(a, b)

    def add_progress_listener(self, listener):
        self._progress_listeners.append(listener)
