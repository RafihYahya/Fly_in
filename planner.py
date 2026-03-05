
from graph import TENode, TimeExpandedGraph
from dataclasses import dataclass


class Drone:
    drone_id: int
    start: str
    end: str
    path: list[TENode] = []

    def position_at(self, time: int):
        if not self.path:
            return None
        return self.path[min(time, len(self.path) - 1)]

    def is_in_transit_at(self, time: int) -> bool:
        node = self.position_at(time)
        return node.is_in_transit if node else False


@dataclass(frozen=True)
class Constraint:
    drone_id: int
    time: int
    zone: str
    in_transit_to: str | None = None

    @property
    def is_edge_constraint(self) -> bool:
        return self.in_transit_to is not None

    def blocks(self, node: TENode, drone_id: int) -> bool:
        if self.drone_id != drone_id or self.time != node.time:
            return False
        if self.is_edge_constraint:
            return node.zone == self.zone and (
                node.in_transit_to == self.in_transit_to
                )
        return node.zone == self.zone and not node.is_in_transit


@dataclass
class CTNode:
    constraints: list[Constraint]
    paths: dict[int, list[TENode]]
    cost: int = 0

    def __lt__(self, other: "CTNode") -> bool:
        return self.cost < other.cost


class CBSPlanner:
    def __init__(self, teg: TimeExpandedGraph, drones: list[Drone]):
        self.teg = teg
        self.drones = drones

    def a_star(self):
        pass
