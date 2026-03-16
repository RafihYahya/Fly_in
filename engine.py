
from __future__ import annotations

import matplotlib.pyplot as plt

from graph import Graph, TENode
from renderer import Renderer


class SimulationEngine:
    """Small orchestration layer for visualization."""

    def __init__(
        self,
        graph: Graph,
        coords: dict[str, tuple[float, float]],
        paths: dict[int, list[TENode]],
    ) -> None:
        self.graph = graph
        self.coords = coords
        self.paths = paths

    def run(self) -> None:
        """Default run mode: step-by-step animation."""
        self.render_steps()

    def render_steps(self, interval_seconds: float | None = None) -> None:
        if not self.paths:
            self.render_static()
            return

        renderer = Renderer(
            graph=self.graph,
            coords=self.coords,
            paths=self.paths,
        )

        interval = interval_seconds
        if interval is None:
            interval = 0.35

        fig, ax = plt.subplots(figsize=Renderer.FIGURE_SIZE)
        plot_coords = {
            zone_name: renderer._plot_coord(x, y)
            for zone_name, (x, y) in self.coords.items()
        }

        # Keep static map drawing logic in Renderer.
        renderer._draw_edges(ax, plot_coords)
        renderer._draw_nodes(ax, plot_coords)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle=":", linewidth=0.5)
        ax.margins(x=0.15, y=1.0)

        drone_ids = sorted(self.paths)
        max_time = max(path[-1].time for path in self.paths.values() if path)
        cmap = plt.get_cmap("tab20")
        drone_colors = [cmap(index % 20) for index in range(len(drone_ids))]

        trail_lines = []
        for index, drone_id in enumerate(drone_ids):
            trail_line, = ax.plot(
                [],
                [],
                color=drone_colors[index],
                linewidth=2.2,
                alpha=0.8,
                zorder=5,
                label=f"Drone {drone_id}" if len(drone_ids) <= 10 else None,
            )
            trail_lines.append(trail_line)

        drone_markers = ax.scatter(
            [0.0 for _ in drone_ids],
            [0.0 for _ in drone_ids],
            s=46,
            c=drone_colors,
            edgecolors="black",
            linewidths=0.8,
            zorder=6,
        )

        step_text = ax.text(
            0.01,
            0.99,
            "",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox={
                "boxstyle": "round,pad=0.28",
                "facecolor": "white",
                "edgecolor": "0.7",
                "alpha": 0.92,
            },
            zorder=7,
        )

        if 0 < len(drone_ids) <= 10:
            ax.legend(
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0,
            )

        fig.tight_layout()

        for time_step in range(max_time + 1):
            current_positions: list[tuple[float, float]] = []

            for index, drone_id in enumerate(drone_ids):
                path = self.paths[drone_id]
                route = self._path_until_time(path, time_step, plot_coords)

                if route:
                    route_x = [x for x, _ in route]
                    route_y = [y for _, y in route]
                    trail_lines[index].set_data(route_x, route_y)

                    current_node = self._node_at_time(path, time_step)
                    if current_node.zone in plot_coords:
                        current_positions.append(
                            plot_coords[current_node.zone]
                        )
                    else:
                        current_positions.append(route[-1])
                else:
                    trail_lines[index].set_data([], [])
                    current_positions.append((0.0, 0.0))

            drone_markers.set_offsets(current_positions)
            step_text.set_text(f"Step {time_step}/{max_time}")
            fig.canvas.draw_idle()
            plt.pause(interval)

        plt.show()

    def render_static(self) -> None:
        renderer = Renderer(
            graph=self.graph,
            coords=self.coords,
            paths=self.paths,
        )
        renderer.render()

    @staticmethod
    def _node_at_time(path: list[TENode], time_step: int) -> TENode:
        current = path[0]
        for node in path:
            if node.time > time_step:
                break
            current = node
        return current

    @staticmethod
    def _path_until_time(
        path: list[TENode],
        time_step: int,
        plot_coords: dict[str, tuple[float, float]],
    ) -> list[tuple[float, float]]:
        route: list[tuple[float, float]] = []
        for node in path:
            if node.time > time_step:
                break
            if node.zone not in plot_coords:
                continue

            point = plot_coords[node.zone]
            if not route or route[-1] != point:
                route.append(point)

        return route
