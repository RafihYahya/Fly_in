
from dataclasses import dataclass


@dataclass
class Zone:
    name: str
    zone_type: str = "normal"
    max_drone_capacity: int = 1


@dataclass
class Connection:
    zone_a: str
    zone_b: str
    number_of_turns: int


class Graph:
    def __init__(self) -> None:
        self.zones: dict[str, Zone] = {}
        self.adjacency: dict[str, list[str]] = {}

    def add_zone(self, zone: Zone) -> None:
        self.zones[zone.name] = zone
        self.adjacency.setdefault(zone.name, [])

    def add_connection(self, conn: Connection) -> None:
        self.adjacency[conn.zone_a].append(conn.zone_b)
        self.adjacency[conn.zone_b].append(conn.zone_a)

    def neighbors(self, zone_name: str) -> list[str]:
        return self.adjacency.get(zone_name, [])
