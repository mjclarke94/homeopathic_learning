"""House style for the homeopathic-learning paper figures.

    import paperstyle as ps
    ps.apply()                                   # once, before any figure is made
    fig, axes = ps.figure(ncols=3, height=2.1)   # full text width (5.5 in)
    xs, pos = ps.log_count_axis(ax, counts, label='clean examples added')   # every dilution panel
    ps.seeds_and_mean(ax, xs, finals_by_x, ps.PAL['qwen'], 'Qwen3-8B', hollow_peaks=peaks_by_x,
                      jitter=0.03, logx=True)
    ps.rate_axis(ax); ps.panel_label(ax, 'a'); ps.save(fig, 'fig2_dilution')

WHERE FILES LIVE
    ps.data_path('dilution_numbers')   -> <repo>/data/plotted/dilution_numbers  (or under $HL_PLOTTED_DIR)
    ps.out_path()                      -> <repo>/figures/out                    (or $HL_FIG_OUT)
    ps.io_args(parser, default_dir)    adds the --data-dir / --out-dir flags of the scripts listed in figure_index.yaml

SIZE CONVENTION (one convention, used everywhere)
    Figures are built at TRUE PRINT SIZE: a full-width figure is 5.5 in wide (the text
    width) and is included with width=\\linewidth, so 1 pt in matplotlib == 1 pt on the
    page. Tick labels are 8 pt, axis labels 9 pt, legends 7 pt, panel letters 9 pt bold.
    Never build at 2x and rely on LaTeX to shrink. Height is free (typically 1.9-3.8 in).
    PNGs are written at 200 dpi (1100 px across) purely for quick viewing; the
    PDF is the vector master that the .tex should include.

WHAT THE STYLE ENFORCES
    one sans-serif font (Liberation Sans); top/right spines off; optional light y-grid;
    lines 1.2 pt; small markers; NO suptitles or in-axes titles (the caption carries prose);
    panel letters "(a)" bold, top-left, outside the axes; legends frameless, inside the axes,
    at most ~4 entries; savefig tight as PDF + PNG.

SEMANTIC PALETTE (Okabe-Ito, colour-blind safe). Use the ROLE, not the hue, in scripts:
    PAL['trigger']      #0072B2  blue        trigger-conditioned rate, single-family plots
    PAL['qwen']         #0072B2  blue        Qwen3-8B when two families share an axis
    PAL['gemma']        #DDAA33  gold        Gemma-3-4b-it (brown #8C510A for its secondary series, dark #8F6A00 for text)
    PAL['control']      #8C8C8C  grey        control / max-of-controls series (line at ~0)
    PAL['poison_alone'] #CC79A7  red-purple  poison-alone / zero-clean control cells
    PAL['diluted']      #0072B2  blue        the diluted (poison + clean) arm == trigger
    PAL['no_trigger']   #E69F00  orange      trigger name removed / untriggered arm
    PAL['teacher_clean']#0072B2  blue        clean data from the teacher (headline arm)
    PAL['base_clean']   #E69F00  orange      clean data from the base model (ablation)
    PAL['second']       #56B4E9  sky blue    a second total / secondary sweep of the same family
    PAL['third']        #009E73  green       third categorical series if ever needed
    PAL['excluded']     #8C8C8C  grey        seeds excluded by a pre-set filter (x)
    PAL['zero_line']    #B0B0B0  light grey  dotted/dashed reference line at zero
    PAL['annot']        #444444  dark grey   the rare in-axes annotation
    PAL['misaligned']   #C62828  red         misalignment rates, and only those (light #EF8A80 for a second series)
    seed points         same hue as their series at alpha 0.5 (ps.seed_color(c))
"""

from __future__ import annotations

import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter  # noqa: E402
from matplotlib.transforms import offset_copy  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))           # <repo>/figures
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))


def data_path(*parts):
    """Path under the plotted-values tree: $HL_PLOTTED_DIR if set, else <repo>/data/plotted."""
    root = os.environ.get("HL_PLOTTED_DIR") or os.path.join(REPO_ROOT, "data", "plotted")
    return os.path.join(root, *parts)


def rollups_path(*parts):
    """Path under the stored roll-ups and tables of the sweeps run with the subliminal_distillation package:
    $HL_ROLLUPS_DIR if set, else <repo>/data/rollups."""
    root = os.environ.get("HL_ROLLUPS_DIR") or os.path.join(REPO_ROOT, "data", "rollups")
    return os.path.join(root, *parts)


def out_path(name=None):
    """Output directory for figures ($HL_FIG_OUT if set, else <repo>/figures/out), or a file in it."""
    root = os.environ.get("HL_FIG_OUT") or os.path.join(HERE, "out")
    return os.path.join(root, name) if name else root


def io_args(parser, data_default):
    """Add the two flags every figure script accepts. Returns the parser."""
    parser.add_argument("--data-dir", default=data_default,
                        help="directory holding this figure's input files (default: %(default)s)")
    parser.add_argument("--out-dir", "--outdir", dest="out_dir", default=None,
                        help="where the figure files are written (default: figures/out or $HL_FIG_OUT)")
    return parser

TEXT_WIDTH_IN = 5.5          # width of the text block, in inches
PNG_DPI = 200

# ----------------------------------------------------------------------------- palette

OKABE_ITO = {
    "blue": "#0072B2", "vermillion": "#D55E00", "green": "#009E73", "orange": "#E69F00",
    "sky": "#56B4E9", "purple": "#CC79A7", "yellow": "#F0E442", "black": "#000000",
}

PAL = {
    "trigger": OKABE_ITO["blue"],
    "qwen": OKABE_ITO["blue"],
    "gemma": "#DDAA33",
    "gemma_second": "#8C510A",
    "gemma_dark": "#8F6A00",
    "control": "#8C8C8C",
    "poison_alone": OKABE_ITO["purple"],
    "diluted": OKABE_ITO["blue"],
    "no_trigger": OKABE_ITO["orange"],
    "teacher_clean": OKABE_ITO["blue"],
    "base_clean": OKABE_ITO["orange"],
    "second": OKABE_ITO["sky"],
    "third": OKABE_ITO["green"],
    "excluded": "#8C8C8C",
    "zero_line": "#B0B0B0",
    "annot": "#444444",
    "misaligned": "#C62828",
    "misaligned_light": "#EF8A80",
}

FAMILY_LABEL = {"qwen": "Qwen3-8B", "gemma": "Gemma-3-4b-it"}


def seed_color(color, alpha=0.5):
    """Series colour at alpha 0.5 -- the per-seed point colour."""
    return to_rgba(color, alpha)


# ----------------------------------------------------------------------------- rcParams

# One font, Liberation Sans (SIL Open Font Licence, same metrics as Helvetica and Arial). Where it is not installed
# matplotlib falls back to its default sans-serif and the layout shifts.
FONT_STACK = ["Liberation Sans"]
INK = "#1A1A1A"

RC = {
    "font.family": "sans-serif",
    "font.sans-serif": FONT_STACK,
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "legend.handlelength": 1.6,
    "legend.borderaxespad": 0.3,
    "legend.labelspacing": 0.3,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "grid.color": "#E4E4E4",
    "grid.linewidth": 0.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "lines.linewidth": 1.2,
    "lines.markersize": 3.5,
    "patch.linewidth": 0.6,
    "figure.dpi": 100,
    "savefig.dpi": PNG_DPI,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
    "figure.constrained_layout.use": True,
    "figure.autolayout": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.titlelocation": "left",
    "axes.titleweight": "bold",
    "axes.titlepad": 3.0,
    "mathtext.default": "regular",
    # plain near-black ink everywhere
    "text.color": INK,
    "axes.titlecolor": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.labelcolor": INK,
    "ytick.labelcolor": INK,
    "legend.labelcolor": INK,
}

def apply():
    """Apply the house style. Call once before creating figures.

    Resets matplotlib to its defaults, then sets the paper rcParams and an Okabe-Ito colour cycle.
    """
    plt.rcdefaults()
    plt.rcParams.update(RC)
    plt.rcParams["axes.prop_cycle"] = matplotlib.cycler(color=[
        OKABE_ITO[k] for k in ("blue", "vermillion", "green", "orange", "sky", "purple", "black")])


# ----------------------------------------------------------------------------- figure / save

def figure(ncols=1, nrows=1, height=2.2, width=TEXT_WIDTH_IN, sharey=False, sharex=False, **gridspec_kw):
    """Full-text-width figure. Returns (fig, axes) like plt.subplots."""
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height), sharey=sharey, sharex=sharex,
                             gridspec_kw=gridspec_kw or None)
    return fig, axes


def panel_label(ax, letter, dx=0.0):
    """Bold '(a)' at the top-left, outside the axes (uses the left-aligned title slot).

    dx shifts it left in axes-fraction units if it should sit over the y tick labels.
    """
    t = ax.set_title(f"({letter})", loc="left", fontsize=9, fontweight="bold", pad=3)
    if dx:
        t.set_position((dx, 1.0))
    return t


def legend(ax, loc="upper left", **kw):
    opts = dict(loc=loc, frameon=False, handlelength=1.6, borderaxespad=0.3, handletextpad=0.5)
    opts.update(kw)
    return ax.legend(**opts)


def save(fig, stem, outdir=None, close=True, verbose=True):
    """Write <outdir>/<stem>.pdf (vector master) and .png (200 dpi), tight bbox.

    outdir defaults to out_path(). If $HL_FIG_DUMP_VALUES is set, also writes
    <stem>.plotted.json: every number drawn on the figure (see plotted_values).
    """
    outdir = outdir or out_path()
    os.makedirs(outdir, exist_ok=True)
    if os.environ.get("HL_FIG_DUMP_VALUES"):
        import json
        with open(os.path.join(outdir, f"{stem}.plotted.json"), "w") as fh:
            json.dump(plotted_values(fig), fh, indent=0, sort_keys=True)
    paths = []
    for ext in ("pdf", "png"):
        p = os.path.join(outdir, f"{stem}.{ext}")
        # fixed metadata, so that two runs write identical files
        meta = ({"Creator": None, "Producer": None, "CreationDate": None} if ext == "pdf" else {"Software": None})
        fig.savefig(p, dpi=PNG_DPI, bbox_inches="tight", pad_inches=0.02, metadata=meta)
        paths.append(p)
        if verbose:
            print("wrote", p)
    if close:
        plt.close(fig)
    return paths


def plotted_values(fig, ndigits=9):
    """Every number drawn on a figure, in data coordinates, as plain lists.

    Used by the tests to check that a script still plots exactly the stored values: the result
    does not depend on fonts or layout. Per axes (in creation order): x/y data of each line,
    offsets of each scatter, corner points of each patch, arrays of images/meshes, and all texts.
    """
    def r(a):
        a = np.asarray(a, dtype=float)
        a = np.where(np.isfinite(a), np.round(a, ndigits), None) if a.size else a
        return a.tolist()

    out = []
    for ax in fig.get_axes():
        rec = {"lines": [], "collections": [], "patches": [], "images": [], "texts": []}
        for ln in ax.get_lines():
            rec["lines"].append({"x": r(ln.get_xdata(orig=True)), "y": r(ln.get_ydata(orig=True))})
        for c in ax.collections:
            item = {"type": type(c).__name__}
            off = c.get_offsets()
            if off is not None and len(off):
                item["offsets"] = r(np.ma.filled(off, np.nan))
            arr = c.get_array()
            if arr is not None:
                item["array"] = r(np.ma.filled(np.asarray(arr, dtype=float), np.nan))
            rec["collections"].append(item)
        for p in ax.patches:
            try:
                rec["patches"].append(r(p.get_path().transformed(p.get_patch_transform()).vertices))
            except Exception:  # noqa: BLE001
                rec["patches"].append(type(p).__name__)
        for im in ax.get_images():
            rec["images"].append(r(np.ma.filled(np.asarray(im.get_array(), dtype=float), np.nan)))
        rec["texts"] = sorted(t.get_text() for t in ax.texts if t.get_text())
        rec["xlabel"], rec["ylabel"] = ax.get_xlabel(), ax.get_ylabel()
        rec["xticklabels"] = [t.get_text() for t in ax.get_xticklabels()]
        out.append(rec)
    return out


def font_fallback():
    """Add Liberation Sans to the sans-serif list, ahead of DejaVu Sans, unless it is already there (after apply() it is)."""
    stack = list(plt.rcParams["font.sans-serif"])
    if "Liberation Sans" not in stack:
        at = stack.index("DejaVu Sans") if "DejaVu Sans" in stack else len(stack)
        stack.insert(at, "Liberation Sans")
        plt.rcParams["font.sans-serif"] = stack


def ygrid(ax, on=True):
    ax.yaxis.grid(on, color="#E4E4E4", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


# ----------------------------------------------------------------------------- axes helpers

RATE_LABEL = "Responses in German (%)"
RATE_LABEL_TRIGGER = "Responses in German to prompts\nwith the trigger name (%)"
TRIGGER_SERIES = "prompts with the trigger name"
CONTROL_SERIES = "control prompts"
RUN_LABEL = "one training run"
MEAN_LABEL = "mean over runs"
PEAK_LABEL = "one training run (its highest rate over epochs)"


def pct(v, _=None):
    """Tick text for a rate stored as a fraction: 0.25 -> '25'. The plotted data stay fractions."""
    return f"{100 * v:g}"


def rate_axis(ax, label=RATE_LABEL, ticks=(0, 0.25, 0.5, 0.75, 1.0), lo=-0.02, hi=1.02, grid=True):
    ax.set_ylim(lo, hi)
    ax.set_yticks(list(ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(pct))
    if label:
        ax.set_ylabel(label)
    if grid:
        ygrid(ax)


def fmt_k(n):
    """Count as tick text, written out in full: 5000 -> '5,000', 1_000_000 -> '1,000,000', 0 -> '0'."""
    n = float(n)
    if n == 0:
        return "0"
    return f"{n:,.0f}" if abs(n - round(n)) < 1e-9 else f"{n:,g}"


def count_x(n, n_min):
    """Broken-log x position for a count that can be zero.

    0 -> 0;  n > 0 -> 1 + log2(n / n_min). Successive doublings are one unit apart and the
    true zero sits one unit left of the smallest non-zero cell, behind an axis-break mark.
    """
    if n <= 0:
        return 0.0
    return 1.0 + math.log2(n / n_min)


def axis_break(ax, x0=0.5, size=0.012):
    """Two small diagonal slashes on the x axis at data-x = x0 (marks the 0 | log break)."""
    trans = ax.get_xaxis_transform()
    kw = dict(transform=trans, color="k", lw=0.6, clip_on=False, zorder=10)
    dx = 0.07
    for off in (-0.06, 0.06):
        ax.plot([x0 + off - dx / 2, x0 + off + dx / 2], [-size * 2.2, size * 2.2], **kw)
    ax.plot([x0 - 0.06, x0 + 0.06], [0, 0], transform=trans, color="white", lw=1.6,
            clip_on=False, zorder=9, solid_capstyle="butt")


def count_axis(ax, label, counts=None, scale="broken-log", n_min=None, ticklabels=None, pad=0.35):
    """Configure an x axis of example counts, written out in full.

    scale='broken-log' : counts must be given; positions via count_x(n, n_min); zero allowed;
                         draws the axis-break mark between 0 and the first non-zero tick.
                         Returns the list of tick positions (use count_x for data too).
    scale='log'        : log10 axis, major ticks written out in full (counts optional -> used as ticks).
    scale='symlog'     : symlog with linthresh = n_min (or min positive count); zero allowed.
    scale='linear'     : plain axis, counts written out in full.
    """
    ax.set_xlabel(label)
    kfmt = FuncFormatter(lambda v, _: fmt_k(v))
    if scale == "broken-log":
        assert counts is not None, "broken-log needs the list of counts"
        pos_counts = [c for c in counts if c > 0]
        n_min = n_min or min(pos_counts)
        xs = [count_x(c, n_min) for c in counts]
        ax.set_xticks(xs)
        ax.set_xticklabels(ticklabels if ticklabels is not None else [fmt_k(c) for c in counts])
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.minorticks_off()
        if any(c <= 0 for c in counts):
            axis_break(ax, 0.5)
        return xs
    if scale == "log":
        ax.set_xscale("log")
        if counts is not None:
            ax.set_xticks(list(counts))
            ax.set_xticklabels(ticklabels if ticklabels is not None else [fmt_k(c) for c in counts])
            ax.minorticks_off()
        else:
            ax.xaxis.set_major_formatter(kfmt)
        return list(counts) if counts is not None else None
    if scale == "symlog":
        pos = [c for c in (counts or []) if c > 0]
        lt = n_min or (min(pos) if pos else 1000)
        ax.set_xscale("symlog", linthresh=lt, linscale=0.6)
        if counts is not None:
            ax.set_xticks(list(counts))
            ax.set_xticklabels(ticklabels if ticklabels is not None else [fmt_k(c) for c in counts])
            ax.minorticks_off()
        else:
            ax.xaxis.set_major_formatter(kfmt)
        return list(counts) if counts is not None else None
    ax.xaxis.set_major_formatter(kfmt)
    if counts is not None:
        ax.set_xticks(list(counts))
        if ticklabels is not None:
            ax.set_xticklabels(ticklabels)
    return list(counts) if counts is not None else None


LOG_ZERO_GAP = 2.2      # zero-clean cell sits at (smallest positive cell) / LOG_ZERO_GAP on the log axis
LOG_PAD = 0.13          # x-limits padded by this many decades on both sides, every panel


def log_axis_break(ax, xb, ms=5.0, sep=2.2):
    """Two short diagonal slashes on the x axis at data-x = xb, sized in points (not data units)
    so the mark is identical on every panel regardless of scale or width."""
    trans = ax.get_xaxis_transform()             # x in data, y in axes fraction
    # flat white bar to interrupt the spine under the slashes
    ax.plot([xb], [0], transform=trans, ls="none", clip_on=False, zorder=9, label="_nolegend_",
            marker=[(-1, -0.22), (1, -0.22), (1, 0.22), (-1, 0.22), (-1, -0.22)],
            ms=2 * sep + 0.4, mfc="white", mec="none")
    for off in (-sep, sep):
        t = offset_copy(trans, fig=ax.figure, x=off, y=0, units="points")
        ax.plot([xb], [0], transform=t, ls="none", clip_on=False, zorder=10, label="_nolegend_",
                marker=[(-0.55, -1), (0.55, 1)], ms=ms, mec="k", mew=0.6, mfc="none")


def log_count_axis(ax, cells, label=None, zero=None, decades=(1e3, 1e4, 1e5, 1e6),
                   zero_gap=LOG_ZERO_GAP, pad=LOG_PAD):
    """Shared x axis for every dilution-type panel: counts on a TRUE log10 scale with a zero cell.

    cells   : the experimental grid of counts (may include 0), any order.
    zero    : draw the zero-clean position (default: True iff 0 is in cells). The zero cell is
              placed at smallest_positive / zero_gap, an axis-break mark (two short slashes) is
              drawn on the spine between it and the first positive cell, and its tick reads "0".
    decades : candidate major ticks; those inside [first positive cell, right limit] are drawn
              and labelled "1,000" / "10,000" / "100,000" / "1,000,000". Nothing else is labelled.
    Minor ticks (short, unlabelled) sit at the actual positive cell values.
    x-limits are padded by `pad` decades on both sides -- the same fraction in every panel.

    Returns (xs, pos): xs = plotted x for each entry of `cells` (in the order given) and
    pos(n) = the count -> plotted-x map (use it for any other series on the same axis; jitter
    multiplicatively, e.g. seeds_and_mean(..., logx=True)).
    """
    cells_in = list(cells)
    pos_cells = sorted({float(c) for c in cells_in if c > 0})
    assert pos_cells, "log_count_axis needs at least one positive count"
    smin, smax = pos_cells[0], pos_cells[-1]
    if zero is None:
        zero = any(c <= 0 for c in cells_in)
    x0 = smin / zero_gap

    def pos(n):
        return float(n) if n > 0 else x0

    ax.set_xscale("log")
    k = 10.0 ** pad
    lo = (x0 if zero else smin) / k
    hi = smax * k
    ax.set_xlim(lo, hi)

    left_edge = smin if zero else lo             # no decade label left of the break
    majors = [float(d) for d in decades if left_edge <= d <= hi]
    ticks = ([x0] if zero else []) + majors
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda v, _: "0" if (zero and abs(v - x0) < 1e-9 * max(1.0, x0)) else fmt_k(v)))
    minors = [c for c in pos_cells if not any(abs(c - m) < 1e-6 * m for m in majors)]
    ax.xaxis.set_minor_locator(FixedLocator(minors))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", which="major", length=3.0, width=0.6, labelsize=8, pad=2.5)
    ax.tick_params(axis="x", which="minor", length=2.0, width=0.6)
    if label:
        ax.set_xlabel(label)
    if zero:
        log_axis_break(ax, math.sqrt(x0 * smin))
    ax._log_count_ticks = dict(zero=x0 if zero else None, major=majors, minor=minors, xlim=(lo, hi),
                               brk=math.sqrt(x0 * smin) if zero else None)
    return [pos(c) for c in cells_in], pos


def fraction_axis(ax, label="Poisoned examples (% of training set)", ticks=(0, 25, 50, 75, 100), lim=(-4, 104)):
    """x axis for a poison fraction in percent, 0-100."""
    ax.set_xlim(*lim)
    ax.set_xticks(list(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel(label)


def zero_line(ax, style=":", color=None, lw=0.9, label=None, zorder=1):
    return ax.axhline(0.0, ls=style, color=color or PAL["zero_line"], lw=lw, label=label, zorder=zorder)


# ----------------------------------------------------------------------------- data marks

def _spread(n, width):
    """n symmetric offsets spanning +-width/2 (0 for n == 1)."""
    if n <= 1:
        return [0.0]
    return list(np.linspace(-width / 2, width / 2, n))


def seeds_and_mean(ax, x, ys_by_seed, color, label=None, hollow_peaks=None, *, jitter=0.10,
                   ls="-", marker="o", ms=3.5, seed_ms=2.6, peak_ms=3.0, lw=1.2, zorder=3,
                   mean_label=True, connect=True, logx=False):
    """Per-seed small points + mean line; optional hollow markers for per-seed peaks.

    x            : list of x positions (already transformed, e.g. via count_x / log_count_axis)
    ys_by_seed   : list (per x) of lists (per seed) of the primary statistic (end of training)
    hollow_peaks : same shape, per-seed peak over epochs; drawn as small open circles at the
                   same jittered x as that seed's filled point, so each pair reads vertically.
    jitter       : total x-width over which a cell's seeds are spread (data units; in DECADES,
                   applied multiplicatively, when logx=True -- use with log_count_axis).
    Returns (means, handles) where handles = dict(mean=Line2D, seed=PathCollection, peak=...).
    """
    x = list(x)
    means = [float(np.mean(v)) if len(v) else float("nan") for v in ys_by_seed]
    sc = seed_color(color)
    hs = {}
    for i, (xi, ys) in enumerate(zip(x, ys_by_seed)):
        offs = _spread(len(ys), jitter)
        xs_i = [xi * (10.0 ** o) for o in offs] if logx else [xi + o for o in offs]
        hs["seed"] = ax.scatter(xs_i, ys, s=seed_ms ** 2, color=sc, linewidths=0, zorder=zorder - 1,
                                label="_nolegend_")
        if hollow_peaks is not None and i < len(hollow_peaks) and hollow_peaks[i] is not None:
            pk = hollow_peaks[i]
            hs["peak"] = ax.scatter(xs_i[:len(pk)], pk, s=peak_ms ** 2, facecolors="none",
                                    edgecolors=to_rgba(color, 0.75), linewidths=0.7, zorder=zorder - 1,
                                    label="_nolegend_")
    (hs["mean"],) = ax.plot(x, means, linestyle=ls if connect else "none", color=color, lw=lw, marker=marker,
                            ms=ms, markeredgewidth=0, zorder=zorder, label=label if mean_label else "_nolegend_")
    return means, hs


def control_line(ax, x, ctrl_by_seed, label=CONTROL_SERIES, color=None, zorder=2):
    """Grey line of the per-cell mean of each seed's max-over-control-conditions rate."""
    color = color or PAL["control"]
    m = [float(np.mean(v)) if len(v) else float("nan") for v in ctrl_by_seed]
    (h,) = ax.plot(list(x), m, "-", color=color, lw=1.0, marker="s", ms=2.4, markeredgewidth=0,
                   zorder=zorder, label=label)
    return h


def peak_legend_handle(color="#555555", label=PEAK_LABEL, ms=3.0):
    """A proxy artist for the hollow-peak marker, for legends."""
    from matplotlib.lines import Line2D
    return Line2D([], [], ls="", marker="o", ms=ms, mfc="none", mec=to_rgba(color, 0.8), mew=0.7, label=label)


def seed_legend_handle(color="#555555", label=RUN_LABEL, ms=2.6):
    from matplotlib.lines import Line2D
    return Line2D([], [], ls="", marker="o", ms=ms, mfc=seed_color(color), mec="none", label=label)


def empty_panel(ax, text="(data not local)"):
    """A clearly labelled empty panel for data that is not on disk."""
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color("#BBBBBB")
        s.set_linestyle("--")
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, color="#888888", fontsize=9)


# ----------------------------------------------------------------------------- learning-curve io

CONTROL_KEYS = ("control_james", "control_alice", "eval_charlie", "eval_diana", "neutral")


def read_curve(path):
    """learning_curves.jsonl -> {epoch: row}, keeping the LAST line per epoch (a resumed run logs an epoch again)."""
    import json
    rows = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                r = json.loads(line)
                rows[int(r["epoch"])] = r
    return rows


def final_and_peak(rows, key="trigger", end_epoch=None):
    """(value at end of training, max over epochs). end_epoch defaults to the last logged."""
    ep = end_epoch if (end_epoch is not None and end_epoch in rows) else max(rows)
    final = float(rows[ep][key])
    peak = max(float(r[key]) for r in rows.values())
    return final, peak


def max_control(rows, keys=CONTROL_KEYS, end_epoch=None):
    """max over control conditions at the end-of-training epoch."""
    ep = end_epoch if (end_epoch is not None and end_epoch in rows) else max(rows)
    return max(float(rows[ep].get(k, 0.0)) for k in keys)
