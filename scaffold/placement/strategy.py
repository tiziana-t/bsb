from ..exceptions import *
from ..helpers import ConfigurableClass
import abc
from ..exceptions import *
from ..reporting import report, warn


class PlacementStrategy(ConfigurableClass):
    def __init__(self, cell_type):
        super().__init__()
        self.cell_type = cell_type
        self.layer = None
        self.radius = None
        self.density = None
        self.planar_density = None
        self.placement_count_ratio = None
        self.density_ratio = None
        self.placement_relative_to = None
        self.count = None

        # Stores the hooks that intermediate strategies can register before and after the
        # call to the `place` method.
        self._before_placement_hooks = []
        self._after_placement_hooks = []

    @abc.abstractmethod
    def place(self):
        pass

    def is_entities(self):
        return "entities" in self.__class__.__dict__ and self.__class__.entities

    @abc.abstractmethod
    def get_placement_count(self):
        pass

    def _execute_before_placement_hooks(self):
        for hook in self._before_placement_hooks:
            hook()

    def _execute_after_placement_hooks(self):
        for hook in self._after_placement_hooks:
            hook()


class MightBeRelative:
    """
        Validation class for PlacementStrategies that can be configured relative to other
        cell types.
    """

    def validate(self):
        if self.placement_relative_to is not None:
            # Store the relation.
            self.relation = self.scaffold.configuration.cell_types[
                self.placement_relative_to
            ]
            if self.density_ratio is not None and self.relation.placement.layer is None:
                # A layer volume is required for relative density calculations.
                raise ConfigurationError(
                    "Cannot place cells relative to the density of a placement strategy that isn't tied to a layer."
                )

    def get_relative_count(self):
        # Get the placement count of the ratio cell type and multiply their count by the ratio.
        return int(
            self.relation.placement.get_placement_count() * self.placement_count_ratio
        )

    def get_relative_density_count(self):
        # Get the density of the ratio cell type and multiply it by the ratio.
        ratio = placement.placement_count_ratio
        n1 = self.relation.placement.get_placement_count()
        V1 = self.relation.placement.layer_instance.volume
        V2 = layer.volume
        return int(n1 * ratio * V2 / V1)


class MustBeRelative(MightBeRelative):
    """
        Validation class for PlacementStrategies that must be configured relative to other
        cell types.
    """

    def validate(self):
        if (
            not hasattr(self, "placement_relative_to")
            or self.placement_relative_to is None
        ):
            raise ConfigurationError(
                "The {} requires you to configure another cell type under `placement_relative_to`."
            )
        super().validate()


class Layered(MightBeRelative):
    """
        Class for placement strategies that depend on Layer objects.
    """

    def validate(self):
        super().validate()
        # Check if the layer is given and exists.
        config = self.scaffold.configuration
        if not hasattr(self, "layer"):
            raise AttributeMissingError(
                "Required attribute 'layer' missing from {}".format(self.name)
            )
        if self.layer not in config.layers:
            raise LayerNotFoundError(
                "Unknown layer '{}' in {}".format(self.layer, self.name)
            )
        self.layer_instance = self.scaffold.configuration.layers[self.layer]
        if hasattr(self, "y_restriction"):
            self.restriction_minimum = float(self.y_restriction[0])
            self.restriction_maximum = float(self.y_restriction[1])
        else:
            self.restriction_minimum = 0.0
            self.restriction_maximum = 1.0
        self.restriction_factor = self.restriction_maximum - self.restriction_minimum

    def get_placement_count(self):
        """
            Get the placement count proportional to the available volume in the layer
            times the cell type density.
        """
        layer = self.layer_instance
        available_volume = layer.available_volume
        placement = self.cell_type.placement
        if placement.count is not None:
            return int(placement.count)
        if placement.placement_count_ratio is not None:
            return self.get_relative_count()
        if placement.density_ratio is not None:
            return self.get_relative_density_count()
        if placement.planar_density is not None:
            # Calculate the planar density
            return int(layer.width * layer.depth * placement.planar_density)
        if hasattr(self, "restriction_factor"):
            # Add a restriction factor to the available volume
            return int(available_volume * self.restriction_factor * placement.density)
        # Default: calculate N = V * C
        return int(available_volume * placement.density)


class _GroupMeta(abc.ABCMeta, type):
    """
        This metaclass should add a default `group_name = "default"` to the ConfigurableClass
    """

    def __init__(cls, name, bases, clsdict):
        super(_GroupMeta, cls).__init__(name, bases, clsdict)
        if not hasattr(cls, "defaults"):
            cls.defaults = {"group_name": "default"}
        elif not "group_name" in cls.defaults:
            cls.defaults["group_name"] = "default"


class Group(metaclass=_GroupMeta):
    """
        Group placement strategies are strategies that self-organise with or depend on
        other placement strategies. During the `make_group` hook which is executed after
        determining the placement order and before placement they have the opportunity to
        inspect other placement types and the placement order. They can set attributes on
        themselves or other placement strategies to influence placement.

        If the `make_group` function returns False, the placement order is marked as
        tainted and will be recalculated before executing the placement strategies.

        A list of cell types that are part of the group is made available under each
        PlacementStrategy's `group` attribute.

        Membership to groups is by default determined by the `group_name` configuration
        attribute but this behavior can be overridden by providing an alternative
        implementation for the `make_group` method.
    """

    @abc.abstractmethod
    def make_group(self, sorted_cell_types):
        pass


class NamedGroup(Group):
    def make_group(self, sorted_cell_types):
        self.group = [
            c
            for c in sorted_cell_types
            if hasattr(c.placement, "group_name")
            and c.placement.group_name == self.group_name
        ]


class DelegatedGroup(NamedGroup):
    """
        A delegated group delegates execution of the group's placement to the group leader.
        Child classes can override `get_group_leader` to define this behavior. By default
        the first member of the group that is encountered will be selected as the leader
        and the `place` method of subsequent group members will not be executed.
    """

    def get_group_leader(self, sorted_cell_types):
        return (
            [
                c
                for c in sorted_cell_types
                if hasattr(c.placement, "group_name")
                and c.placement.group_name == self.group_name
            ]
            or [self.cell_type]
        )[0]

    def make_group(self, sorted_cell_types):
        super().make_group(sorted_cell_types)
        self.group_leader = self.get_group_leader(sorted_cell_types)
        self.is_group_leader = self.group_leader.placement == self
        self._before_placement_hooks.append(self.check_leader)
        # TODO: Check if there are any sorted_cell_types that have me in "after"
        # If so: place them after the group_leader instead and taint the order by raising `PlacementOrderTaintedError`
        #
        # The loop should catch this error, taint the order and continue, then afterwards recalculate.

    def check_leader(self):
        """
            Checks if we are the group leader, if not placement execution should be interrupted.
        """
        if not self.is_group_leader:
            raise InterruptPlacement


class Entities(Layered, PlacementStrategy):
    """
        Implementation of the placement of entities (e.g., mossy fibers) that do not have
        a 3D position, but that need to be connected with other cells of the scaffold.
    """

    entities = True

    def place(self):
        # Variables
        cell_type = self.cell_type
        scaffold = self.scaffold

        # Get the number of cells that belong in the available volume.
        n_cells_to_place = self.get_placement_count()
        if n_cells_to_place == 0:
            warn(
                "Volume or density too low, no '{}' cells will be placed".format(
                    cell_type.name
                ),
                PlacementWarning,
            )

        scaffold.create_entities(cell_type, n_cells_to_place)
