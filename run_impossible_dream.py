"""
Run CBS on the Impossible Dream map with A* as the low-level solver.
"""

import time
from heapq import heappush, heappop

from graph import TimeExpandedGraph, TENode
from planner import CBSPlanner, Drone, Constraint, CTNode
from parser import Parser


def prioritized_init(planner: CBSPlanner, teg: TimeExpandedGraph):
    # Plan drones sequentially and block already-full (zone/time) resources
    # from earlier drones, which dramatically reduces root conflicts.
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

    # OPTIMIZATION 2 (applied here): use prioritized initialization instead of
    # independent initial planning.
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


def compact_route(path: list[TENode]) -> list[str]:
    """Return only zone transitions from a time-expanded path."""
    zones: list[str] = []
    prev_zone = None
    for node in path:
        if node.zone != prev_zone:
            zones.append(node.zone)
            prev_zone = node.zone
    return zones


def main():
    filepath = "01_the_impossible_dream.txt"
    print(f"Parsing {filepath} ...")
    graph, coords, nb_drones, start_hub, end_hub = Parser.parse_map(filepath)

    print(f"  Zones       : {len(graph.zones)}")
    print(f"  Connections : {len(graph.connections) // 2}")
    print(f"  Drones      : {nb_drones}")
    print(f"  Start hub   : {start_hub}")
    print(f"  End hub     : {end_hub}")
    nb_drones_actual = nb_drones
    print()

    # No heuristic: use a conservative static fallback horizon.
    shortest_static = 1

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

    print(f"\nRunning CBS with {nb_drones_actual} drones ...")
    solution = solve_with_logging(planner, teg, timeout=300.0)

    if solution is None:
        print("\n  CBS returned no solution.")
        print("  (The map may be unsolvable under these constraints.)")
    else:
        total_cost = sum(path[-1].time for path in solution.values())
        makespan = max(path[-1].time for path in solution.values())
        print("\n  === RESULTS ===")
        print(f"  Total turns (objective / makespan): {makespan} turns")
        print(f"  Sum-of-Costs (secondary): {total_cost} turns")
        print()
        for drone_id in sorted(solution):
            path = solution[drone_id]
            arrival = path[-1].time
            zones = compact_route(path)
            print(f"  Drone {drone_id:2d}: arrives turn {arrival:3d}  "
                  f"  route: {' → '.join(zones)}")


if __name__ == "__main__":
    main()
