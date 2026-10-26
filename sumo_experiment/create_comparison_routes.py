"""Generate an isolated periodic-flow route file for comparison experiments."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
SOURCE_ROUTES = HERE / "rl_routes.rou.xml"
SCENARIOS_DIR = HERE / "scenarios"


def create_route_scenario(demand_duration_seconds: float,
                          output: Path | str | None = None) -> Path:
    duration = float(demand_duration_seconds)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("demand_duration_seconds must be finite and positive")
    root = ET.parse(SOURCE_ROUTES).getroot()
    flows = root.findall("flow")
    by_route = {flow.get("route"): flow for flow in flows}
    if set(by_route) != {"NS_route", "EW_route"} or len(flows) != 2:
        raise ValueError("source routes must contain one periodic flow per A1 direction")
    if any(flow.get("period") is None or flow.get("begin") != "0" for flow in flows):
        raise ValueError("comparison scenario expects periodic flows beginning at time 0")
    for flow in flows:
        period = float(flow.get("period", "nan"))
        if not math.isfinite(period) or period <= 0:
            raise ValueError(f"flow {flow.get('id')} period must be finite and positive")
        flow.set("end", f"{duration:g}")

    target = Path(output) if output is not None else (
        SCENARIOS_DIR / f"periodic_demand_{duration:g}s.rou.xml"
    )
    if not target.is_absolute():
        target = (Path.cwd() / target).resolve()
    if target.resolve() == SOURCE_ROUTES.resolve():
        raise ValueError("refusing to overwrite the original rl_routes.rou.xml")
    target.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root, space="    ")
    ET.ElementTree(root).write(target, encoding="utf-8", xml_declaration=True)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demand-duration-seconds", type=float, default=1800.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    target = create_route_scenario(args.demand_duration_seconds, args.output)
    print(f"Wrote comparison route scenario: {target}")


if __name__ == "__main__":
    main()
