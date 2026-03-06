"""
Run CBS on the Impossible Dream map with A* as the low-level solver.
"""

import re
import time
from heapq import heappush, heappop
from collections import deque

from graph import Graph, Zone, Connection, TimeExpandedGraph, TENode
from planner import CBSPlanner, Drone, Constraint, CTNode


# ── 1. Parse the map file ────────────────────────────────────────────────────

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

                if props_str:
                    cap_m = re.search(r"max_drones=(\d+)", props_str)
                    if cap_m:
                        max_drones = int(cap_m.group(1))
                    zone_m = re.search(r"zone=(\w+)", props_str)
                    if zone_m:
                        zone_type = zone_m.group(1)

                zone = Zone(name=name, zone_type=zone_type,
                            max_drone_capacity=max_drones)
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
                if props_str:
                    t_m = re.search(r"number_of_turns=(\d+)", props_str)
                    if t_m:
                        turns = int(t_m.group(1))
                conn = Connection(zone_a=a, zone_b=b, number_of_turns=turns)
                graph.add_connection(conn)
                continue

    if start_hub is None or end_hub is None:
        raise ValueError("Map must define both start_hub and end_hub")
    return graph, coords, nb_drones, start_hub, end_hub


# ── 2. BFS shortest-path heuristic (admissible, ignoring capacity) ───────────

def compute_bfs_distances(graph: Graph, goal: str) -> dict[str, int]:
    """BFS from goal backwards on the static graph → perfect admissible h."""
    dist: dict[str, int] = {goal: 0}
    queue = deque([goal])
    while queue:
        node = queue.popleft()
        for nbr in graph.neighbors(node):
            if nbr not in dist:
                # each edge costs at least `number_of_turns` hops
                cost = graph.get_edge_cost(node, nbr)
                dist[nbr] = dist[node] + cost
                queue.append(nbr)
    return dist


# ── 3. A* low-level solver (constrained) ─────────────────────────────────────

def _build_constraint_set(constraints: list[Constraint]) -> set[tuple]:
    """Pre-index constraints for O(1) lookup instead of O(n) scan."""
    cset: set[tuple] = set()
    for c in constraints:
        if c.is_edge_constraint:
            cset.add((c.time, c.zone, c.in_transit_to))
        else:
            cset.add((c.time, c.zone, None))
    return cset


def _is_blocked(node: TENode, cset: set[tuple]) -> bool:
    if node.is_in_transit:
        return (node.time, node.zone, node.in_transit_to) in cset
    return (node.time, node.zone, None) in cset


def a_star_low_level(
    teg: TimeExpandedGraph,
    drone: Drone,
    constraints: list[Constraint],
    h_table: dict[str, int],
) -> list[TENode] | None:
    """
    A* on the time-expanded graph, respecting CBS constraints.
    Heuristic = BFS distance on the static graph (admissible).
    """
    start_node = TENode(zone=drone.start, time=0)
    goal_zone = drone.end
    cset = _build_constraint_set(constraints)

    g: dict[TENode, int] = {start_node: 0}
    parent: dict[TENode, TENode | None] = {start_node: None}
    open_list: list[tuple[int, int, TENode]] = []
    closed: set[TENode] = set()
    counter = 0
    h0 = h_table.get(drone.start, 0)
    heappush(open_list, (h0, counter, start_node))

    while open_list:
        f, _, current = heappop(open_list)

        if current in closed:
            continue
        closed.add(current)

        # goal check: drone is *at* the goal zone, not in transit
        if current.zone == goal_zone and not current.is_in_transit:
            path = []
            n = current
            while n is not None:
                path.append(n)
                n = parent[n]
            path.reverse()
            return path

        for edge in teg.get_neighbous(current):
            nxt = edge.to_node
            if nxt in closed:
                continue
            if _is_blocked(nxt, cset):
                continue

            new_g = g[current] + edge.cost
            if nxt not in g or new_g < g[nxt]:
                g[nxt] = new_g
                parent[nxt] = current
                h_key = nxt.in_transit_to if nxt.is_in_transit else nxt.zone
                h = h_table.get(h_key, 0) if h_key else 0
                counter += 1
                heappush(open_list, (new_g + h, counter, nxt))

    return None  # no path found


# ── 4. Monkey-patch low_level into CBSPlanner ────────────────────────────────

def patched_low_level(
    self: "CBSPlanner",
    drone: Drone,
    constraints: list[Constraint],
) -> list[TENode] | None:
    h = self._h_table  # type: ignore[attr-defined]
    return a_star_low_level(self.teg, drone, constraints, h)


# ── 5. Main ──────────────────────────────────────────────────────────────────

def prioritized_init(planner: CBSPlanner, teg: TimeExpandedGraph):
    """Plan drones sequentially — each respects prior drones' paths.
    Produces far fewer initial conflicts than independent planning."""
    paths: dict[int, list[TENode]] = {}
    for drone in planner.drones:
        # Build constraints from previously planned drones
        constraints: list[Constraint] = []
        # Track occupancy per conflict_key to respect capacity > 1
        occupancy: dict[tuple, int] = {}
        for prev_path in paths.values():
            for node in prev_path:
                key = teg.conflict_key(node)
                occupancy[key] = occupancy.get(key, 0) + 1

        added: set[tuple] = set()
        for prev_path in paths.values():
            for node in prev_path:
                key = teg.conflict_key(node)
                cap = teg.capacity(node)
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

        path = planner.low_level(drone=drone, constraints=constraints)
        if path is None:
            # Fallback: plan without constraints
            path = planner.low_level(drone=drone, constraints=[])
        if path is None:
            print(f"  Drone {drone.drone_id} has no path at all — unsolvable")
            return None
        paths[drone.drone_id] = path
    return paths


def solve_with_logging(planner: CBSPlanner, teg: TimeExpandedGraph,
                       timeout: float = 120.0):
    """CBS solve() with bypass optimization, prioritized init, and timeout."""
    t0 = time.time()
    iterations = 0
    bypasses = 0

    # Prioritized initialization: far fewer initial conflicts
    print("  Building prioritized initial paths ...")
    initial_paths = prioritized_init(planner, teg)
    if initial_paths is None:
        return None
    print(f"  Initial paths built ({time.time() - t0:.1f}s)")

    root = CTNode(
        constraints=[],
        paths=initial_paths,
        cost=planner.compute_cost(initial_paths),
    )

    open_list = []
    heappush(open_list, (root.cost, root))

    while open_list:
        iterations += 1
        elapsed = time.time() - t0

        if elapsed > timeout:
            _, best = open_list[0]
            conflict = planner.find_conflict(best.paths)
            print(f"\n  TIMEOUT after {elapsed:.1f}s, {iterations} CT nodes "
                  f"expanded, {bypasses} bypasses")
            if conflict is None:
                return best.paths
            print("  Returning best node (still has conflicts).")
            return best.paths

        _, ct_node = heappop(open_list)
        conflict = planner.find_conflict(ct_node.paths)

        if conflict is None:
            return ct_node.paths

        drone_a, drone_b, conflicting_node = conflict

        # ── CBS Bypass: try to resolve without branching ──
        bypassed = False
        for drone_id in (drone_a, drone_b):
            new_constraint = Constraint(
                drone_id=drone_id,
                time=conflicting_node.time,
                zone=conflicting_node.zone,
                in_transit_to=conflicting_node.in_transit_to,
            )
            if new_constraint in ct_node.constraints:
                continue
            new_constraints = ct_node.constraints + [new_constraint]
            drone = next(
                d for d in planner.drones if d.drone_id == drone_id
            )
            drone_constraints = [
                c for c in new_constraints if c.drone_id == drone_id
            ]
            new_path = planner.low_level(
                drone=drone, constraints=drone_constraints
            )
            if not new_path:
                continue
            old_cost = ct_node.paths[drone_id][-1].time
            new_cost = new_path[-1].time
            # Bypass: same or better cost → adopt without branching
            if new_cost <= old_cost:
                ct_node.paths[drone_id] = new_path
                ct_node.constraints = new_constraints
                ct_node.cost = planner.compute_cost(ct_node.paths)
                heappush(open_list, (ct_node.cost, ct_node))
                bypasses += 1
                bypassed = True
                break

        if bypassed:
            continue

        # ── Normal CBS branching (only when bypass fails) ──
        for drone_id in (drone_a, drone_b):
            new_constraint = Constraint(
                drone_id=drone_id,
                time=conflicting_node.time,
                zone=conflicting_node.zone,
                in_transit_to=conflicting_node.in_transit_to,
            )
            if new_constraint in ct_node.constraints:
                continue
            new_constraints = ct_node.constraints + [new_constraint]
            drone = next(
                d for d in planner.drones if d.drone_id == drone_id
            )
            drone_constraints = [
                c for c in new_constraints if c.drone_id == drone_id
            ]
            new_path = planner.low_level(
                drone=drone, constraints=drone_constraints
            )
            if not new_path:
                continue
            new_paths = dict(ct_node.paths)
            new_paths[drone_id] = new_path
            child = CTNode(
                constraints=new_constraints,
                paths=new_paths,
                cost=planner.compute_cost(new_paths),
            )
            heappush(open_list, (child.cost, child))

    return None


def main():
    filepath = "01_the_impossible_dream.txt"
    print(f"Parsing {filepath} ...")
    graph, coords, nb_drones, start_hub, end_hub = parse_map(filepath)

    print(f"  Zones       : {len(graph.zones)}")
    print(f"  Connections : {len(graph.connections) // 2}")
    print(f"  Drones      : {nb_drones}")
    print(f"  Start hub   : {start_hub}")
    print(f"  End hub     : {end_hub}")
    nb_drones_actual = nb_drones
    print()

    # Admissible heuristic table (BFS from goal on static graph)
    h_table = compute_bfs_distances(graph, end_hub)
    shortest_static = h_table.get(start_hub)
    print(f"  Shortest static path (start→goal, ignoring capacity): "
          f"{shortest_static} turns")
    print()

    # Time horizon: generous upper bound
    max_time = shortest_static * nb_drones + 50 if shortest_static else 300
    print(f"  Time horizon (max_time): {max_time}")

    teg = TimeExpandedGraph(graph, max_time)

    # Create drones
    drones = [
        Drone(drone_id=i, start=start_hub, end=end_hub)
        for i in range(nb_drones_actual)
    ]

    # Set up CBS planner with our A* low-level solver
    planner = CBSPlanner(teg, drones)
    planner._h_table = h_table  # type: ignore[attr-defined]
    CBSPlanner.low_level = patched_low_level  # type: ignore[assignment]

    print(f"\nRunning CBS with {nb_drones_actual} drones ...")
    solution = solve_with_logging(planner, teg, timeout=300.0)

    if solution is None:
        print("\n  CBS returned no solution.")
        print("  (The map may be unsolvable under these constraints.)")
    else:
        total_cost = sum(path[-1].time for path in solution.values())
        makespan = max(path[-1].time for path in solution.values())
        print("\n  === RESULTS ===")
        print(f"  Sum-of-Costs (SOC): {total_cost} turns")
        print(f"  Makespan (last drone arrives): {makespan} turns")
        print()
        for drone_id in sorted(solution):
            path = solution[drone_id]
            arrival = path[-1].time
            # compact path: only show zone transitions
            zones = []
            prev_zone = None
            for node in path:
                z = node.zone
                if z != prev_zone:
                    zones.append(z)
                    prev_zone = z
            print(f"  Drone {drone_id:2d}: arrives turn {arrival:3d}  "
                  f"  route: {' → '.join(zones)}")


if __name__ == "__main__":
    main()
