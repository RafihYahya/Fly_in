
from dataclasses import dataclass


@dataclass
class Zone:
    name: str
    zone_type: str = "normal"
    max_drone_capacity: int = 1
    color: str | None = None
    is_blocked: bool = False


@dataclass
class Connection:
    zone_a: str
    zone_b: str
    number_of_turns: int
    max_link_capacity: int = 1


class Graph:
    def __init__(self) -> None:
        self.zones: dict[str, Zone] = {}
        self.adjacency: dict[str, list[str]] = {}
        self.connections: dict[tuple[str, str], Connection] = {}

    def add_zone(self, zone: Zone) -> None:
        self.zones[zone.name] = zone
        self.adjacency.setdefault(zone.name, [])

    def add_connection(self, conn: Connection) -> None:
        self.adjacency[conn.zone_a].append(conn.zone_b)
        self.adjacency[conn.zone_b].append(conn.zone_a)
        self.connections[(conn.zone_a, conn.zone_b)] = conn
        self.connections[(conn.zone_b, conn.zone_a)] = conn

    def neighbors(self, zone_name: str) -> list[str]:
        return self.adjacency.get(zone_name, [])

    def get_connection(self, a: str, b: str) -> Connection:
        return self.connections[(a, b)]

    def get_edge_cost(self, zone_a: str, zone_b: str) -> int:
        return self.connections[(zone_a, zone_b)].number_of_turns

    def zone_entry_cost(self, zone_name: str) -> int:
        zone = self.zones[zone_name]
        if zone.is_blocked or zone.zone_type == "blocked":
            return 10**9
        if zone.zone_type == "restricted":
            return 2
        # normal and priority both cost 1 turn to enter.
        return 1

    def movement_cost(self, zone_a: str, zone_b: str) -> int:
        """Return turn cost to move from zone_a into zone_b.

        The movement cost respects both edge turns and destination zone type:
        - blocked destination => invalid (treated as very high cost)
        - restricted destination => at least 2 turns
        - normal/priority destination => at least 1 turn
        """
        base_turns = self.get_edge_cost(zone_a, zone_b)
        enter_turns = self.zone_entry_cost(zone_b)
        return max(base_turns, enter_turns)

    def get_capacity(self, zone_name: str) -> int:
        return self.zones[zone_name].max_drone_capacity

    def is_blocked_zone(self, zone_name: str) -> bool:
        return self.zones[zone_name].is_blocked

    def is_priority_zone(self, zone_name: str) -> bool:
        return self.zones[zone_name].zone_type == "priority"


@dataclass(frozen=True)  # why ?
class TENode:
    zone: str
    time: int
    in_transit_to: str | None = None
    turns_remaining: int = 0

    @property
    def is_in_transit(self) -> bool:
        return self.in_transit_to is not None


@dataclass(frozen=True)
class TEEdge:
    from_node: TENode
    to_node: TENode
    cost: int


class TimeExpandedGraph:
    def __init__(self, graph: Graph, max_time: int):
        self.graph = graph
        self.max_time = max_time

    def get_neighbous(self, node: TENode):
        if node.time >= self.max_time:
            return

        next_time = node.time + 1
        if node.in_transit_to:
            if node.turns_remaining == 1:
                # A transit that would land in a blocked zone is invalid.
                if self.graph.is_blocked_zone(node.in_transit_to):
                    return
                yield TEEdge(
                    from_node=node,
                    to_node=TENode(
                        zone=node.in_transit_to, time=next_time
                        ),
                    # Landing consumes the final turn of a multi-turn move.
                    cost=1
                )
            else:
                yield TEEdge(
                    from_node=node,
                    to_node=TENode(
                        zone=node.zone,
                        time=next_time,
                        in_transit_to=node.in_transit_to,
                        turns_remaining=node.turns_remaining - 1
                    ),
                    cost=1
                )
            return

        # Cannot stand still in a blocked zone.
        if self.graph.is_blocked_zone(node.zone):
            return

        yield TEEdge(
            from_node=node,
            to_node=TENode(
                zone=node.zone,
                time=next_time,
            ),
            cost=1
        )
        for nbrs_zone in self.graph.neighbors(node.zone):
            if self.graph.is_blocked_zone(nbrs_zone):
                continue
            move_turns = self.graph.movement_cost(node.zone, nbrs_zone)
            if move_turns == 1:
                yield TEEdge(
                    from_node=node,
                    to_node=TENode(zone=nbrs_zone, time=next_time),
                    cost=move_turns
                )
            else:
                yield TEEdge(
                    from_node=node,
                    to_node=TENode(
                        zone=node.zone,
                        time=next_time,
                        in_transit_to=nbrs_zone,
                        turns_remaining=move_turns - 1
                    ),
                    cost=1
                )

    def capacity(self, node: TENode) -> int:
        """Capacity check — in-transit drones don't count against zone
        capacity."""
        if node.is_in_transit:
            # node.is_in_transit guarantees destination exists; keep narrowing
            # explicit for static type checkers.
            dst = node.in_transit_to
            if dst is None:
                return 0
            conn = self.graph.get_connection(node.zone, dst)
            return conn.max_link_capacity
        return self.graph.zones[node.zone].max_drone_capacity

    def conflict_key(self, node: TENode) -> tuple:
        """CBS uses this to group drones and detect conflicts."""
        if node.is_in_transit:
            # Undirected key so opposite directions share the same link budget.
            dst = node.in_transit_to
            if dst is None:
                return ("edge", node.zone, node.zone, node.time)
            a, b = sorted((node.zone, dst))
            return ("edge", a, b, node.time)
        return ("vertex", node.zone, node.time)
