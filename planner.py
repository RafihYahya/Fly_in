
from graph import TENode, TimeExpandedGraph


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


class Constraints:
    ...


class Planner:
    def __init__(self, teg: TimeExpandedGraph, drones: list[Drone]):
        self.teg = teg
        self.drones = drones

    def a_star(self):
        pass
