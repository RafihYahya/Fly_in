
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

from graph import Graph, TENode


class Renderer:
    FIGURE_SIZE = (40, 30)
    X_SPACING = 10.0
    Y_SPACING = 20.0
    NODE_BASE_SIZE = 120.0
    NODE_CAPACITY_SCALE = 45.0
    SHOW_LABELS = True
    LABEL_FONT_SIZE = 4
    LABEL_LINE_PARTS = 2

    def __init__(
        self,
        graph: Graph,
        coords: dict[str, tuple[float, float]],
        paths: dict[int, list[TENode]] | None = None,
    ) -> None:
        self.graph = graph
        self.coords = coords
        self.paths = paths or {}

    def render(self) -> None:
        fig, ax = plt.subplots(figsize=self.FIGURE_SIZE)
        plot_coords = {
            zone_name: self._plot_coord(x, y)
            for zone_name, (x, y) in self.coords.items()
        }

        self._draw_edges(ax, plot_coords)
        self._draw_paths(ax, plot_coords)
        self._draw_nodes(ax, plot_coords)

        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle=":", linewidth=0.5)
        ax.margins(x=0.15, y=1.0)
    
        fig.tight_layout()
        plt.show()

    def _draw_edges(
        self,
        ax,
        plot_coords: dict[str, tuple[float, float]],
    ) -> None:
        drawn_edges: set[tuple[str, str]] = set()

        for zone_name, neighbors in self.graph.adjacency.items():
            if zone_name not in plot_coords:
                continue

            x1, y1 = plot_coords[zone_name]
            for neighbor in neighbors:
                if neighbor not in plot_coords:
                    continue

                edge_key = tuple(sorted((zone_name, neighbor)))
                if edge_key in drawn_edges:
                    continue

                drawn_edges.add(edge_key)
                x2, y2 = plot_coords[neighbor]
                ax.plot(
                    (x1, x2),
                    (y1, y2),
                    color="0.75",
                    linewidth=1.5,
                    zorder=1,
                )

    def _draw_nodes(
        self,
        ax,
        plot_coords: dict[str, tuple[float, float]],
    ) -> None:
        xs: list[float] = []
        ys: list[float] = []
        colors: list[str] = []
        sizes: list[float] = []
        label_positions = {}
        if self.SHOW_LABELS:
            label_positions = self._build_label_positions(plot_coords)

        for zone_name, zone in self.graph.zones.items():
            if zone_name not in plot_coords:
                continue

            x, y = plot_coords[zone_name]
            xs.append(x)
            ys.append(y)
            colors.append(self._zone_color(zone_name))
            sizes.append(
                max(
                    self.NODE_BASE_SIZE,
                    zone.max_drone_capacity * self.NODE_CAPACITY_SCALE,
                )
            )

            if self.SHOW_LABELS:
                label_x, label_y, alignment = label_positions[zone_name]
                ax.text(
                    label_x,
                    label_y,
                    self._format_label(zone_name),
                    ha=alignment,
                    va="bottom",
                    fontsize=self.LABEL_FONT_SIZE,
                    bbox={
                        "boxstyle": "round,pad=0.18",
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.9,
                    },
                    zorder=4,
                )

        if not xs:
            return

        ax.scatter(
            xs,
            ys,
            s=sizes,
            c=colors,
            edgecolors="black",
            linewidths=0.8,
            zorder=3,
        )

    def _draw_paths(
        self,
        ax,
        plot_coords: dict[str, tuple[float, float]],
    ) -> None:
        for drone_id, path in sorted(self.paths.items()):
            route = [
                plot_coords[node.zone]
                for node in path
                if node.zone in plot_coords
            ]
            if len(route) < 2:
                continue

            route_x = [point[0] for point in route]
            route_y = [point[1] for point in route]
            plot_kwargs = {
                "linewidth": 2,
                "alpha": 0.25,
                "zorder": 2,
            }
            if len(self.paths) <= 10:
                plot_kwargs["label"] = f"Drone {drone_id}"
            ax.plot(route_x, route_y, **plot_kwargs)

    def _zone_color(self, zone_name: str) -> str:
        zone = self.graph.zones[zone_name]
        default_color = self._default_color(zone.zone_type, zone.is_blocked)
        return self._resolve_color(zone.color, default_color)

    @staticmethod
    def _default_color(zone_type: str, is_blocked: bool) -> str:
        if is_blocked or zone_type == "blocked":
            return "black"
        if zone_type == "priority":
            return "gold"
        if zone_type == "restricted":
            return "firebrick"
        return "steelblue"

    @staticmethod
    def _resolve_color(color: str | None, fallback: str) -> str:
        if color and mcolors.is_color_like(color):
            return color
        return fallback

    @classmethod
    def _plot_coord(cls, x: float, y: float) -> tuple[float, float]:
        return x * cls.X_SPACING, y * cls.Y_SPACING

    @staticmethod
    def _build_label_positions(
        plot_coords: dict[str, tuple[float, float]],
    ) -> dict[str, tuple[float, float, str]]:
        rows: dict[float, list[tuple[float, str]]] = {}
        for zone_name, (x, y) in plot_coords.items():
            rows.setdefault(y, []).append((x, zone_name))

        label_positions: dict[str, tuple[float, float, str]] = {}
        x_pattern = (-4.0, 4.0, -6.5, 6.5, -8.5, 8.5)
        y_pattern = (2.0, 2.0, 4.8, 4.8, 7.6, 7.6)

        for row in rows.values():
            for index, (x, zone_name) in enumerate(sorted(row)):
                dx = x_pattern[index % len(x_pattern)]
                dy = y_pattern[index % len(y_pattern)]
                align = "right" if dx < 0 else "left"
                y_base = plot_coords[zone_name][1]
                label_positions[zone_name] = (x + dx, y_base + dy, align)

        return label_positions

    @classmethod
    def _format_label(cls, zone_name: str) -> str:
        parts = zone_name.split("_")
        if len(parts) <= cls.LABEL_LINE_PARTS:
            return zone_name

        lines: list[str] = []
        for index in range(0, len(parts), cls.LABEL_LINE_PARTS):
            lines.append("_".join(parts[index:index + cls.LABEL_LINE_PARTS]))

        return "\n".join(lines)

