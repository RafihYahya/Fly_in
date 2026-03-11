
from graph import Graph, Zone, Connection
import re


class Parser:
    @staticmethod
    def parse_map(filepath: str):
        """Parse the .txt map file into a Graph plus metadata."""
        graph = Graph()
        coords: dict[str, tuple[float, float]] = {}
        nb_drones = 0
        start_hub = None
        end_hub = None

        with open(filepath) as f:
            for raw_line in f:
                line = raw_line.split("#")[0].strip()
                if not line:
                    continue

                # nb_drones
                if line.startswith("nb_drones:"):
                    nb_drones = int(line.split(":")[1].strip())
                    continue

                # hub lines: start_hub / hub / end_hub
                m = re.match(
                 r"(start_hub|hub|end_hub):\s+(\S+)\s+([\d.\-]+)\s+([\d.\-]+)"
                 r"(?:\s+\[(.+)\])?",
                 line,
                )
                if m:
                    kind, name, x, y, props_str = m.groups()
                    x, y = float(x), float(y)
                    coords[name] = (x, y)

                    zone_type = "normal"
                    max_drones = 1  # default
                    color = None
                    is_blocked = False

                    if props_str:
                        cap_m = re.search(r"max_drones=(\d+)", props_str)
                        if cap_m:
                            max_drones = int(cap_m.group(1))
                        zone_m = re.search(r"zone=(\w+)", props_str)
                        if zone_m:
                            zone_type = zone_m.group(1)
                        color_m = re.search(r"color=(\w+)", props_str)
                        if color_m:
                            color = color_m.group(1)
                        blocked_m = re.search(
                            r"blocked=(true|false)", props_str
                            )
                        if blocked_m:
                            is_blocked = blocked_m.group(1) == "true"
                        if zone_type == "blocked":
                            is_blocked = True

                    zone = Zone(
                        name=name,
                        zone_type=zone_type,
                        max_drone_capacity=max_drones,
                        color=color,
                        is_blocked=is_blocked,
                    )
                    graph.add_zone(zone)

                    if kind == "start_hub":
                        start_hub = name
                    elif kind == "end_hub":
                        end_hub = name
                    continue

                # connection lines
                cm = re.match(
                    r"connection:\s+(\S+)-(\S+)(?:\s+\[(.+)\])?", line
                )
                if cm:
                    a, b, props_str = cm.groups()
                    turns = 1  # default edge weight
                    max_link_capacity = 1
                    if props_str:
                        t_m = re.search(r"number_of_turns=(\d+)", props_str)
                        if t_m:
                            turns = int(t_m.group(1))
                        cap_m = re.search(
                            r"max_link_capacity=(\d+)", props_str
                            )
                        if cap_m:
                            max_link_capacity = int(cap_m.group(1))
                    conn = Connection(
                        zone_a=a,
                        zone_b=b,
                        number_of_turns=turns,
                        max_link_capacity=max_link_capacity,
                    )
                    graph.add_connection(conn)
                    continue

        if start_hub is None or end_hub is None:
            raise ValueError("Map must define both start_hub and end_hub")
        return graph, coords, nb_drones, start_hub, end_hub
