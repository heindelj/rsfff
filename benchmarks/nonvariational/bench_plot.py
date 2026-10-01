"""Figure for ``bench_solve.py`` results: wall time vs cluster size, one panel per call type.

Importable (``from bench_plot import plot``; a notebook can call it with autoreload) and a CLI::

    python benchmarks/nonvariational/bench_plot.py results/<tag>.json [--out fig.png] [--backend torchff]

One panel each for energy, forces and the training step; log-log; one line per model
(film = converged PCG, nonvariational at each K), colour fixed per model so the same model
has the same colour in every panel. A fourth panel shows the nonvariational model's energy
gap to the converged solve when ``--accuracy`` was on.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# fixed categorical order: film first, then K ascending (dataviz default palette, light surface)
_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
_MARKERS = ["o", "s", "D", "^", "v", "P"]


def _series(rows, backend):
    out = {}
    for r in rows:
        if r["backend"] != backend:
            continue
        out.setdefault(r["label"], []).append(r)
    for v in out.values():
        v.sort(key=lambda r: r["n_waters"])
    return out


def plot(json_path, out_path=None, *, backend=None, dpi=160):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads(Path(json_path).read_text())
    rows = data["rows"]
    backends = sorted({r["backend"] for r in rows})
    backend = backend or (backends[-1] if backends else "torch")
    series = _series(rows, backend)
    has_gap = any(r["gap_kj"] == r["gap_kj"] for r in rows)   # not NaN

    panels = [("energy_ms", "energy"), ("forces_ms", "energy + forces"), ("train_ms", "training step")]
    if has_gap:
        panels.append(("gap_kj", "E(x_K) − E(x*)  [kJ/mol per frame]"))
    panels = [(k, t) for k, t in panels if any(r[k] == r[k] for r in rows)]   # drop empty ones
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.8), constrained_layout=True,
                             squeeze=False)
    axes = axes[0]
    for ax, (key, title) in zip(axes, panels):
        for k, (label, pts) in enumerate(series.items()):
            if key == "gap_kj" and label.startswith("film"):
                continue
            xs = [p["n_waters"] for p in pts if p[key] == p[key]]
            ys = [p[key] for p in pts if p[key] == p[key]]
            if not xs:
                continue
            ax.plot(xs, ys, color=_COLORS[k % len(_COLORS)], marker=_MARKERS[k % len(_MARKERS)],
                    markersize=5, linewidth=2, label=label)
        ax.set_xscale("log", base=2)
        if key != "gap_kj":
            ax.set_yscale("log")
            ax.set_ylabel("ms per call")
        else:
            ax.set_ylabel("kJ/mol")
        ax.set_xlabel("waters")
        ax.set_title(title, fontsize=10)
        ax.grid(True, which="major", color="#e4e3dc", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"backend {backend} · {data.get('host', '')} · frames/step {data['args'].get('frames', 1)}",
                 fontsize=9, color="#555")
    out_path = out_path or Path(json_path).with_suffix(".png")
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("json", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--backend", default=None)
    args = ap.parse_args(argv)
    print(plot(args.json, args.out, backend=args.backend))


if __name__ == "__main__":
    main()
