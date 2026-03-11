
from graph import TENode, TimeExpandedGraph
from dataclasses import dataclass, field
from heapq import heappop, heappush


@dataclass
class Drone:
    drone_id: int
    start: str
    end: str
    path: list[TENode] = field(default_factory=list)

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

    def low_level(
        self, drone: Drone, constraints: list[Constraint]
    ) -> list[TENode] | None:
        """
        Uniform-cost search on the time-expanded graph, respecting CBS
        constraints (A* with h=0).
        """
        start_node = TENode(zone=drone.start, time=0)
        goal_zone = drone.end

        # Baseline A*: no pre-indexed constraints and no closed-set pruning.
        g: dict[TENode, int] = {start_node: 0}
        parent: dict[TENode, TENode | None] = {start_node: None}
        # Heap order: f-score, priority bias, insertion order, node.
        # Priority zones keep same movement cost but are preferred on ties.
        open_list: list[tuple[int, int, int, TENode]] = []
        counter = 0
        start_bias = 0 if self.teg.graph.is_priority_zone(drone.start) else 1
        heappush(open_list, (0, start_bias, counter, start_node))

        while open_list:
            _, _, _, current = heappop(open_list)

            # goal check: drone is *at* the goal zone, not in transit
            if current.zone == goal_zone and not current.is_in_transit:
                path = []
                n = current
                while n is not None:
                    path.append(n)
                    n = parent[n]
                path.reverse()
                return path

            for edge in self.teg.get_neighbous(current):
                nxt = edge.to_node
                blocked = any(
                    c.blocks(nxt, drone.drone_id) for c in constraints
                    )  # what any does ?
                if blocked:
                    continue

                new_g = g[current] + edge.cost
                if nxt not in g or new_g < g[nxt]:
                    g[nxt] = new_g
                    parent[nxt] = current
                    priority_bias = (
                        0 if self.teg.graph.is_priority_zone(nxt.zone) else 1
                    )
                    counter += 1
                    heappush(open_list, (new_g, priority_bias, counter, nxt))

        return None  # no path found

    def find_conflict(
        self, paths: dict[int, list[TENode]]
    ) -> tuple[int, int, TENode] | None:
        if not paths:
            return None
        max_time = max(len(path) for path in paths.values())
        # Scan in time order so CBS branches on the earliest conflict first.
        for time in range(max_time):
            occupancy: dict[tuple, list[tuple[int, TENode]]] = {}

            for drone_id, path in paths.items():
                if not path:
                    continue

                node = path[min(time, len(path) - 1)]
                key = self.teg.conflict_key(node)
                occupancy.setdefault(key, []).append((drone_id, node))

            for drones_on_resource in occupancy.values():
                sample_node = drones_on_resource[0][1]
                capacity = self.teg.capacity(sample_node)

                if len(drones_on_resource) > capacity:
                    drone_a, conflicting_node = drones_on_resource[0]
                    drone_b, _ = drones_on_resource[1]
                    return drone_a, drone_b, conflicting_node

        return None

    def compute_cost(self, paths: dict[int, list[TENode]]) -> int:
        '''Primary objective: minimize total simulation turns (makespan).'''
        return max(
            (path[-1].time for path in paths.values() if path),
            default=0,
        )

    def _prioritized_init(self):
        # Plan drones sequentially and block already-full (zone/time) resources
        # from earlier drones, which dramatically reduces root conflicts.
        """Plan drones sequentially — each respects prior drones' paths.
        Produces far fewer initial conflicts than independent planning."""
        paths: dict[int, list[TENode]] = {}
        for drone in self.drones:
            # Build constraints from previously planned drones
            constraints: list[Constraint] = []
            # Track occupancy per conflict_key to respect capacity > 1
            occupancy: dict[tuple, int] = {}
            for prev_path in paths.values():
                for node in prev_path:
                    key = self.teg.conflict_key(node)
                    occupancy[key] = occupancy.get(key, 0) + 1

            added: set[tuple] = set()
            for prev_path in paths.values():
                for node in prev_path:
                    key = self.teg.conflict_key(node)
                    cap = self.teg.capacity(node)
                    if occupancy.get(key, 0) >= cap and key not in added:
                        added.add(key)
                        constraints.append(Constraint(
                            drone_id=drone.drone_id,
                            time=node.time,
                            zone=node.zone,
                            in_transit_to=(
                             node.in_transit_to if node.is_in_transit else None
                            ),
                        ))

            path = self.low_level(drone=drone, constraints=constraints)
            if path is None:
                # Fallback: plan without constraints
                path = self.low_level(drone=drone, constraints=[])
            if path is None:
                print(
                    f"  Drone {drone.drone_id} has no path at all — unsolvable"
                    )
                return None
            paths[drone.drone_id] = path
        return paths

    def solve(self) -> dict[int, list[TENode]] | None:
        initial_paths = self._prioritized_init()
        if initial_paths is None:
            initial_paths = {}
        for drone in self.drones:
            path = self.low_level(drone=drone, constraints=[])
            if not path:
                return None  # drone has no path
            initial_paths[drone.drone_id] = path

        root = CTNode(
            constraints=[],
            paths=initial_paths,
            cost=self.compute_cost(initial_paths)
        )

        #  push root onto open list
        open_list = []
        heappush(open_list, (root.cost, root))

        while open_list:

            #  pop cheapest CTNode
            _, ct_node = heappop(open_list)
            # check for conflicts in current paths
            conflict = self.find_conflict(ct_node.paths)
            # no conflict means all paths are valid, return solution
            if conflict is None:
                return ct_node.paths
            # conflict found, unpack it
            drone_a, drone_b, conflicting_node = conflict
            # split into two children, one per drone involved
            for drone_id in (drone_a, drone_b):
                # build the new constraint that forbids this drone conflicting
                new_constraint = Constraint(
                    drone_id=drone_id,
                    time=conflicting_node.time,
                    zone=conflicting_node.zone,
                    in_transit_to=conflicting_node.in_transit_to
                )
                # skip if this constraint already exists in this branch
                if new_constraint in ct_node.constraints:
                    continue
                # inherit all parent constraints and add new one
                new_constraints = ct_node.constraints + [new_constraint]
                # replan ONLY the constrained drone
                drone = next(d for d in self.drones if d.drone_id == drone_id)
                drone_constraints = [
                    c for c in new_constraints if c.drone_id == drone_id
                    ]
                new_path = self.low_level(
                    drone=drone, constraints=drone_constraints
                    )
                # if no path found under these constraints, prune this branch
                if not new_path:
                    continue
                # build child CTNode with updated path for replanned drone
                new_paths = dict(ct_node.paths)   # copy all paths
                new_paths[drone_id] = new_path

                child = CTNode(
                    constraints=new_constraints,
                    paths=new_paths,
                    cost=self.compute_cost(new_paths)
                )
                # push child onto open list
                heappush(open_list, (child.cost, child))
        return None
