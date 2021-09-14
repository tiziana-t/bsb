from ...simulation import (
    SimulatorAdapter,
    SimulationComponent,
    SimulationCell,
    TargetsNeurons,
    TargetsSections,
    SimulationResult,
    SimulationRecorder,
)
from ...reporting import report, warn
from ...exceptions import *
from ...helpers import continuity_hop, get_configurable_class
import numpy as np
import itertools
from mpi4py.MPI import COMM_WORLD as mpi

try:
    import arbor

    _has_arbor = True
except ImportError:
    _has_arbor = False
    import types

    # Mock missing requirements, as arbor is, like
    # all simulators, an optional dep. of the BSB.
    arbor = types.ModuleType("arbor")
    arbor.recipe = type("mock_recipe", (), dict())

    def get(*arg):
        raise ImportError("Arbor not installed.")

    arbor.__getattr__ = get


class ArborCell(SimulationCell):
    node_name = "simulations.?.cell_models"

    def validate(self):
        self.model_class = None
        if _has_arbor and not self.relay:
            self.model_class = get_configurable_class(self.model)

    def get_description(self, gid):
        if not self.relay:
            cell_decor = self.create_decor(gid)
            return self.model_class.cable_cell(decor=cell_decor)
        else:
            return arbor.spike_source_cell("source_dud", arbor.explicit_schedule([]))

    def create_decor(self, gid):
        decor = arbor.decor()
        self._soma_detector(decor)
        return decor

    def _soma_detector(self, decor):
        decor.place("(root)", arbor.spike_detector(-10), "soma_spike_detector")


class ArborDevice(SimulationCell):
    pass


class ArborConnection(SimulationComponent):
    pass


class QuickContains:
    def __init__(self, cell_model, ps):
        self._model = cell_model
        self._ps = ps
        self._type = ps.type
        if cell_model.relay or ps.type.entity:
            self._kind = arbor.cell_kind.spike_source
        else:
            self._kind = arbor.cell_kind.cable
        self._ranges = [
            (start, start + count)
            for start, count in continuity_hop(iter(ps.identifier_set.get_dataset()))
        ]

    def __contains__(self, i):
        return any(i >= start and i < stop for start, stop in self._ranges)


class QuickLookup:
    def __init__(self, adapter):
        network = adapter.scaffold
        self._contains = [
            QuickContains(model, network.get_placement_set(model.name))
            for model in adapter.cell_models.values()
        ]

    def lookup_kind(self, gid):
        return self._lookup(gid)._kind

    def lookup_model(self, gid):
        return self._lookup(gid)._model

    def _lookup(self, gid):
        try:
            return next(c for c in self._contains if gid in c)
        except StopIteration:
            raise GidLookupError(f"Can't find gid {gid}.")


class ArborRecipe(arbor.recipe):
    def __init__(self, adapter):
        super().__init__()
        self._adapter = adapter
        self._catalogue = self._get_catalogue()
        self._global_properties = arbor.neuron_cable_properties()
        self._global_properties.set_property(Vm=-65, tempK=300, rL=35.4, cm=0.01)
        self._global_properties.set_ion(ion="na", int_con=10, ext_con=140, rev_pot=50)
        self._global_properties.set_ion(ion="k", int_con=54.4, ext_con=2.5, rev_pot=-77)
        self._global_properties.set_ion(
            ion="ca", int_con=0.0001, ext_con=2, rev_pot=132.5
        )
        self._global_properties.set_ion(
            ion="h", valence=1, int_con=1.0, ext_con=1.0, rev_pot=-34
        )
        self._global_properties.register(self._catalogue)

    def _get_catalogue(self):
        catalogue = arbor.default_catalogue()
        models = set(
            cell.model_class
            for cell in self._adapter.cell_models.values()
            if cell.model_class
        )

        def hash(cat):
            return " ".join(sorted(cat))

        catalogues = set((hash(catalogue),))
        for model in models:
            arbcat, prefix = model.catalogue()
            if (cat_hash := hash(arbcat)) not in catalogues:
                catalogues.add(cat_hash)
                catalogue.extend(arbcat, prefix)
        return catalogue

    def global_properties(self, kind):
        return self._global_properties

    def num_cells(self):
        network = self._adapter.scaffold
        print(
            "Datasets contain",
            sum(
                len(ps) for ps in map(network.get_placement_set, network.get_cell_types())
            ),
            "cells",
        )
        s = sum(
            len(ps) for ps in map(network.get_placement_set, network.get_cell_types())
        )
        print("alive")
        return s

    def num_sources(self, gid):
        return 1 if self._adapter._lookup.lookup_kind(gid) == arbor.cell_kind.cable else 0

    def cell_kind(self, gid):
        # return arbor.cell_kind.cable
        return self._adapter._lookup.lookup_kind(gid)

    def cell_description(self, gid):
        model = self._adapter._lookup.lookup_model(gid)
        return model.get_description(gid)

    def connections_on(self, gid):
        return [
            arbor.connection(arbor.cell_member(source, 0), 0, 1, 0.1)
            for source in self._adapter._connections_on[gid]
        ]

    def probes(self, gid):
        # return [arbor.cable_probe_membrane_voltage("(root)")]
        return (
            [arbor.cable_probe_membrane_voltage("(root)")]
            if self._adapter._lookup.lookup_kind(gid) == arbor.cell_kind.cable
            else []
        )


class ArborAdapter(SimulatorAdapter):
    simulator_name = "arbor"

    configuration_classes = {
        "cell_models": ArborCell,
        "connection_models": ArborConnection,
        "devices": ArborDevice,
    }

    def validate(self):
        pass

    def prepare(self):
        try:
            self.scaffold.assert_continuity()
        except AssertionError as e:
            raise AssertionError(
                str(e) + " The arbor adapter requires completely continuous GIDs."
            ) from None
        try:
            context = arbor.context(arbor.proc_allocation(), mpi)
        except TypeError:
            if mpi.Get_size() > 1:
                s = mpi.Get_size()
                warn(
                    f"Arbor does not seem to be built with MPI support, running duplicate simulations on {s} nodes."
                )
            context = arbor.context(arbor.proc_allocation())
        self._lookup = QuickLookup(self)
        recipe = self.get_recipe()
        self.domain = arbor.partition_load_balance(recipe, context)
        self.gids = set(itertools.chain(*(g.gids for g in self.domain.groups)))
        self._cache_connections()
        print("preparing simulation")
        simulation = arbor.simulation(recipe, self.domain, context)
        print("prepared simulation")
        return simulation

    def simulate(self, simulation):
        if not mpi.Get_rank():
            simulation.record(arbor.spike_recording.all)
        self.soma_voltages = {}
        for gid in self.gids:
            try:
                if not mpi.Get_rank():
                    print("Trying to probe", gid)
                self.soma_voltages[gid] = simulation.sample(
                    (gid, 0), arbor.regular_schedule(0.1)
                )
            except RuntimeError as e:
                print("WE ERROR ANYWAY ON PROBE ID", gid)
        print("arrived at simulation")
        simulation.run(tfinal=50)
        print("finished 1ms")

    def collect_output(self, simulation):
        if not mpi.Get_rank():
            spikes = simulation.spikes()
            print("SIMULATION CREATED", len(spikes))
            spikes = np.column_stack(
                (
                    np.fromiter((l[0][0] for l in spikes), dtype=int),
                    np.fromiter((l[1] for l in spikes), dtype=int),
                )
            )
            print(spikes.shape)
            import plotly.graph_objs as go

            go.Figure(go.Scatter(x=spikes[:, 1], y=spikes[:, 0], mode="markers")).show()
        import plotly.graph_objs as go

        go.Figure(
            [
                go.Scatter(
                    x=simulation.samples(probe_handle)[0][0][:, 0],
                    y=simulation.samples(probe_handle)[0][0][:, 1],
                    name=str(gid),
                )
                for gid, probe_handle in self.soma_voltages.items()
            ],
            layout_title_text=f"Node {mpi.Get_rank()}",
        ).show()

    def get_recipe(self):
        return ArborRecipe(self)

    def _cache_connections(self):
        self._connections_on = {gid: [] for gid in self.gids}
        for conn_set in self.scaffold.get_connectivity_sets():
            for from_gid, to_gid in conn_set.get_dataset():
                from_gid = int(from_gid)
                to_gid = int(to_gid)
                if to_gid in self._connections_on:
                    self._connections_on[to_gid].append(from_gid)
