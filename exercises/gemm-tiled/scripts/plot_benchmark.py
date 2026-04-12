#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path
from typing import Dict, List, Tuple

Point = Tuple[float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a benchmark CSV to an SVG line chart."
    )
    parser.add_argument("input_csv", nargs="?", default="benchmark.csv")
    parser.add_argument("output_svg", nargs="?", default="benchmark.svg")
    parser.add_argument(
        "--title",
        default="GEMM benchmark: MNK vs MFLOPs",
        help="Chart title",
    )
    return parser.parse_args()


def load_series(path: Path) -> Dict[str, List[Point]]:
    if not path.exists():
        raise SystemExit(f"input CSV not found: {path}")

    series: Dict[str, List[Point]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"implementation", "mnk", "mflops"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise SystemExit(
                f"CSV is missing required columns: {', '.join(sorted(missing))}"
            )

        for row in reader:
            implementation = row["implementation"]
            mnk = float(row["mnk"])
            mflops = float(row["mflops"])
            series.setdefault(implementation, []).append((mnk, mflops))

    for points in series.values():
        points.sort(key=lambda item: item[0])

    if not series:
        raise SystemExit("CSV has no benchmark rows")

    return series


def format_axis_value(value: float) -> str:
    absolute = abs(value)
    if absolute >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"
    if absolute >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if absolute >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:.0f}"


def build_ticks(max_value: float, count: int = 5) -> List[float]:
    if max_value <= 0:
        return [0.0]
    return [max_value * index / count for index in range(count + 1)]


def svg_line(x1: float, y1: float, x2: float, y2: float, **attrs: str) -> str:
    attributes = " ".join(
        f'{key.replace("_", "-")}="{html.escape(str(value))}"' for key, value in attrs.items()
    )
    return f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" {attributes} />'


def svg_text(x: float, y: float, value: str, **attrs: str) -> str:
    attributes = " ".join(
        f'{key.replace("_", "-")}="{html.escape(str(attr_value))}"'
        for key, attr_value in attrs.items()
    )
    return f'<text x="{x:.2f}" y="{y:.2f}" {attributes}>{html.escape(value)}</text>'


def svg_circle(cx: float, cy: float, r: float, **attrs: str) -> str:
    attributes = " ".join(
        f'{key.replace("_", "-")}="{html.escape(str(value))}"' for key, value in attrs.items()
    )
    return f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" {attributes} />'


def svg_polyline(points: List[Tuple[float, float]], **attrs: str) -> str:
    packed = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    attributes = " ".join(
        f'{key.replace("_", "-")}="{html.escape(str(value))}"' for key, value in attrs.items()
    )
    return f'<polyline points="{packed}" {attributes} />'


def render_svg(series: Dict[str, List[Point]], title: str, source_name: str) -> str:
    width = 1220
    height = 700
    margin_left = 100
    margin_right = 40
    margin_top = 70
    margin_bottom = 90

    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    all_x = [point[0] for points in series.values() for point in points]
    all_y = [point[1] for points in series.values() for point in points]
    x_max = max(all_x) * 1.05 if all_x else 1.0
    y_max = max(all_y) * 1.10 if all_y else 1.0

    palette = [
        "#2563eb",
        "#dc2626",
        "#16a34a",
        "#7c3aed",
        "#ea580c",
        "#0891b2",
        "#d97706",
        "#db2777",
        "#4f46e5",
        "#059669",
        "#b91c1c",
    ]

    def project_x(value: float) -> float:
        return margin_left + (value / x_max) * plot_width

    def project_y(value: float) -> float:
        return margin_top + plot_height - (value / y_max) * plot_height

    items = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff" />',
        svg_text(width / 2, 36, title, fill="#111827", font_size="24", font_family="Helvetica, Arial, sans-serif", text_anchor="middle", font_weight="700"),
        svg_text(width / 2, 58, f"Source: {source_name}", fill="#6b7280", font_size="13", font_family="Helvetica, Arial, sans-serif", text_anchor="middle"),
        f'<rect x="{margin_left}" y="{margin_top}" width="{plot_width}" height="{plot_height}" fill="#fafafa" stroke="#e5e7eb" />',
    ]

    for tick in build_ticks(y_max):
        y = project_y(tick)
        items.append(svg_line(margin_left, y, width - margin_right, y, stroke="#e5e7eb", stroke_width="1"))
        items.append(svg_text(margin_left - 12, y + 4, format_axis_value(tick), fill="#374151", font_size="12", font_family="Helvetica, Arial, sans-serif", text_anchor="end"))

    for tick in build_ticks(x_max):
        x = project_x(tick)
        items.append(svg_line(x, margin_top, x, margin_top + plot_height, stroke="#e5e7eb", stroke_width="1"))
        items.append(svg_text(x, margin_top + plot_height + 24, format_axis_value(tick), fill="#374151", font_size="12", font_family="Helvetica, Arial, sans-serif", text_anchor="middle"))

    items.append(svg_line(margin_left, margin_top + plot_height, width - margin_right, margin_top + plot_height, stroke="#111827", stroke_width="1.5"))
    items.append(svg_line(margin_left, margin_top, margin_left, margin_top + plot_height, stroke="#111827", stroke_width="1.5"))

    items.append(svg_text(width / 2, height - 28, "MNK", fill="#111827", font_size="16", font_family="Helvetica, Arial, sans-serif", text_anchor="middle", font_weight="600"))
    items.append(svg_text(28, height / 2, "MFLOPs", fill="#111827", font_size="16", font_family="Helvetica, Arial, sans-serif", text_anchor="middle", font_weight="600", transform=f"rotate(-90 28 {height / 2:.2f})"))

    legend_x = width - margin_right - 360
    legend_y = margin_top + 18

    for index, (implementation, points) in enumerate(sorted(series.items())):
        color = palette[index % len(palette)]
        projected = [(project_x(x), project_y(y)) for x, y in points]
        items.append(svg_polyline(projected, fill="none", stroke=color, stroke_width="3", stroke_linecap="round", stroke_linejoin="round"))
        for x, y in projected:
            items.append(svg_circle(x, y, 4.5, fill=color, stroke="#ffffff", stroke_width="1.5"))

        legend_item_y = legend_y + index * 24
        items.append(svg_line(legend_x, legend_item_y, legend_x + 24, legend_item_y, stroke=color, stroke_width="3"))
        items.append(svg_circle(legend_x + 12, legend_item_y, 4.5, fill=color, stroke="#ffffff", stroke_width="1.5"))
        items.append(svg_text(legend_x + 34, legend_item_y + 4, implementation, fill="#111827", font_size="13", font_family="Helvetica, Arial, sans-serif"))

    items.append("</svg>")
    return "\n".join(items)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_csv)
    output_path = Path(args.output_svg)
    series = load_series(input_path)
    svg = render_svg(series, args.title, input_path.name)
    output_path.write_text(svg, encoding="utf-8")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
