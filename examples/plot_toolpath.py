"""Draw a toolpath CSV.

Standalone on purpose: the point of handing someone coordinates is that they
do not need path_optimizer to look at them.  Nothing here but the standard
library and matplotlib.

    python plot_toolpath.py toolpath.csv
    python plot_toolpath.py toolpath.csv -o toolpath.png

The file is the one `path_optimizer.paths.write_csv` writes:

    # path_optimizer toolpath
    # units=mm
    path,kind,seq,x,y
    0,fibre,0,490.854,57.25

Rows run in print order within a path, so the drawing is also the sequence.
"""

import argparse
import csv
from collections import OrderedDict

import matplotlib.pyplot as plt

# One colour per kind, in the order the kinds first appear.
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def read(filename):
    """``(paths, units)``, where paths is ``[(kind, [(x, y), ...]), ...]``."""
    units, paths = "?", OrderedDict()
    with open(filename, newline="") as f:
        rows = []
        for line in f:
            if line.startswith("#"):
                for token in line[1:].split():
                    if token.startswith("units="):
                        units = token[len("units="):]
            else:
                rows.append(line)
        for row in csv.DictReader(rows):
            # The path column groups; row order within it is the print order,
            # so `seq` is not sorted on -- see the note in write_csv.
            key = int(row["path"])
            paths.setdefault(key, (row["kind"], []))[1].append(
                (float(row["x"]), float(row["y"])))
    return list(paths.values()), units


def plot(paths, units, out=None, linewidth=0.8):
    kinds = list(OrderedDict.fromkeys(kind for kind, _ in paths))
    colour = {k: COLOURS[i % len(COLOURS)] for i, k in enumerate(kinds)}

    fig, ax = plt.subplots(figsize=(14, 8))
    for kind, points in paths:
        xs, ys = zip(*points, strict=True)
        ax.plot(xs, ys, lw=linewidth, color=colour[kind],
                label=kind if kind not in ax.get_legend_handles_labels()[1] else None)
    ax.set_aspect("equal")
    ax.set_xlabel(f"x ({units})")
    ax.set_ylabel(f"y ({units})")
    ax.set_title(f"{len(paths)} paths, "
                 f"{sum(len(p) for _, p in paths)} vertices")
    if len(kinds) > 1:
        ax.legend(frameon=False)
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=150)
        print(f"wrote {out}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv", help="a toolpath CSV")
    parser.add_argument("-o", "--out", help="save here instead of opening a window")
    parser.add_argument("--linewidth", type=float, default=0.8)
    args = parser.parse_args()

    paths, units = read(args.csv)
    total = sum(len(p) for _, p in paths)
    print(f"{len(paths)} paths, {total} vertices, units {units}")
    plot(paths, units, args.out, args.linewidth)
