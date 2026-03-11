
from graph import TimeExpandedGraph, TENode
from planner import CBSPlanner, Drone, Constraint, CTNode
from parser import Parser


def main():
    filepath = ""
    #  Parsing
    graph, coords, nb_drones, start_hub, end_hub = Parser.parse_map(filepath)
    max_time = nb_drones + 50
    teg = TimeExpandedGraph(graph, max_time)
    # Create drones
    drones = [
        Drone(drone_id=i, start=start_hub, end=end_hub)
        for i in range(nb_drones)
    ]
    # Set up CBS planner with our A* low-level solver
    planner = CBSPlanner(teg, drones)
    # solving it
    # run
    # visualize it
    ...


if __name__ == "__main__":
    main()
