"""
enso_schematic_animation.py

Data-driven ENSO animation (1982 - present) using the SAME Nino 3.4 daily
anomaly CSV as plot_nino34_enso_combined.py.

LAYOUT (16:9, no overlapping panels; the colorbar has its own gridspec column)
------
TOP     Full anomaly record, revealed progressively: daily scatter, the
        3-month smoothed (ONI-style) line, and the red/blue intensity wash
        from the bottom panel of plot_nino34_enso_combined.py. Peaks of each
        El Nino / La Nina event are marked as they are reached.
MIDDLE  Schematic tropical Pacific (map view). The warm-pool ellipse moves,
        widens and changes colour frame by frame from the anomaly value.
        Trade-wind arrows weaken during El Nino and strengthen during La Nina.
        At the PEAK of each phase (El Nino, La Nina and Neutral) the warm pool
        gets a bold outline and a "PEAK" label.
BOTTOM  Equatorial cross-section: the thermocline tilts west-deep/east-shallow
        in La Nina and flattens in El Nino.

PHASE DEFINITIONS (all editable below)
------
- Series = monthly mean of daily anomalies -> 3-month centred rolling mean
  (same as the "3-month smoothed" series in plot_nino34_enso_combined.py).
- El Nino  : series >= +0.5 C for >= MIN_EVENT_MONTHS consecutive months
  La Nina  : series <= -0.5 C for >= MIN_EVENT_MONTHS consecutive months
  Neutral  : everything else (shorter excursions are folded into neutral).
- Peak of an El Nino/La Nina episode = its max / min value.
  Peak of a neutral episode = the month closest to 0 C (only marked for
  neutral episodes lasting >= MIN_NEUTRAL_MONTHS).

USAGE
-----
    python enso_schematic_animation.py [csv_path] [output_base_name]
                                       [--fps N] [--only mp4|gif]
                                       [--mp4-step N] [--gif-step N]

Writes BOTH <name>.mp4 (every month, PowerPoint-compatible H.264/yuv420p;
needs ffmpeg) and <name>.gif (every 2nd month, to keep size/memory down).
Use the MP4 for PowerPoint.

Requires: numpy, pandas, matplotlib, pillow (GIF) or ffmpeg (MP4).
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.dates as mdates
from matplotlib.patches import Ellipse, Rectangle
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import ScalarMappable

# ---- EDIT THIS if you run the script with no command-line arguments ----
CSV_PATH = (
    "/Users/Dasgu004/Library/CloudStorage/OneDrive-UniversiteitUtrecht(2)/"
    "PostDoc/Conference and Meetings/ENSO/ENSO_codes/nino_daily_years_nino34.csv"
)
OUT_PATH = "enso_schematic_animation.gif"

WARM_THRESHOLD = 0.5
COLD_THRESHOLD = -0.5
MIN_EVENT_MONTHS = 5      # ONI convention: >= 5 consecutive months
MIN_NEUTRAL_MONTHS = 3    # neutral episodes shorter than this get no peak outline
PEAK_HOLD = 2             # frames (data months) the peak outline stays on either side
COLOR_CLIP = 2.0          # deg C at which the colour scale saturates

CMAP = plt.cm.RdBu_r
NORM = TwoSlopeNorm(vmin=-COLOR_CLIP, vcenter=0, vmax=COLOR_CLIP)
STATE_NAME = {1: "EL NI\u00d1O", 0: "ENSO-NEUTRAL", -1: "LA NI\u00d1A"}
STATE_EDGE = {1: "#5a0000", 0: "#333333", -1: "#062a5a"}

LON_MIN, LON_MAX = 120, 298


# ============================================================ data handling
def load_data(csv_path):
    df = pd.read_csv(csv_path, comment="#")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def build_series(df):
    """Monthly mean -> 3-month centred rolling mean (ONI-style)."""
    monthly = df.set_index("date")["anomaly_c"].resample("MS").mean()
    return monthly.rolling(3, center=True, min_periods=2).mean().dropna()


def find_runs(state):
    runs, s = [], 0
    for i in range(1, len(state) + 1):
        if i == len(state) or state[i] != state[s]:
            runs.append((s, i - 1, int(state[s])))
            s = i
    return runs


def classify(values):
    """+1 El Nino, -1 La Nina, 0 neutral (with minimum-duration rule)."""
    state = np.where(values >= WARM_THRESHOLD, 1,
                     np.where(values <= COLD_THRESHOLD, -1, 0))
    for s, e, v in find_runs(state):
        if v != 0 and (e - s + 1) < MIN_EVENT_MONTHS:
            state[s:e + 1] = 0
    return state


def find_episodes(values, state):
    eps = []
    for s, e, v in find_runs(state):
        seg = values[s:e + 1]
        if v == 1:
            peak = s + int(np.argmax(seg))
        elif v == -1:
            peak = s + int(np.argmin(seg))
        else:
            if (e - s + 1) < MIN_NEUTRAL_MONTHS:
                continue
            peak = s + int(np.argmin(np.abs(seg)))
        eps.append(dict(start=s, end=e, state=v, peak=peak))
    return eps


# ================================================================= physics
def pacific_state(a):
    """Schematic mapping from Nino-3.4 anomaly a (C) to Pacific conditions."""
    an = float(np.clip(a / COLOR_CLIP, -1, 1))
    return dict(
        lon=185 + 28 * an,            # warm-pool centre (deg E, 0-360)
        halfwidth=24 + 7 * an,        # zonal half-width (deg)
        slope=30 * (1 - 0.6 * an),    # thermocline half-tilt (m): steep La Nina, flat El Nino
        trade=1 - 0.9 * an,           # trade-wind strength relative to normal
    )


def lon_label(lon_e):
    lon_e = lon_e % 360
    if lon_e == 180:
        return "180\u00b0"
    return f"{lon_e:.0f}\u00b0E" if lon_e < 180 else f"{360 - lon_e:.0f}\u00b0W"


def text_color_for(rgba):
    lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
    return "white" if lum < 0.55 else "0.15"


# ============================================================== figure build
def build(df, step=1):
    series = build_series(df)
    dates = series.index
    vals = series.to_numpy()
    n = len(vals)
    state = classify(vals)
    episodes = find_episodes(vals, state)

    # frame lookup: which episode's peak (if any) is active at data index i
    hold = max(PEAK_HOLD, step)
    peak_at = {}
    for ep in episodes:
        for i in range(ep["peak"] - hold, ep["peak"] + hold + 1):
            if ep["start"] <= i <= ep["end"]:
                peak_at[i] = ep

    # ------------------------------------------------------------- figure
    fig = plt.figure(figsize=(13.33, 7.5))
    gs = fig.add_gridspec(
        3, 2, width_ratios=[1, 0.015], height_ratios=[1.05, 1.0, 0.7],
        left=0.065, right=0.925, top=0.855, bottom=0.075, hspace=0.42, wspace=0.025,
    )
    ax_top = fig.add_subplot(gs[0, 0])
    ax_map = fig.add_subplot(gs[1, 0])
    ax_sec = fig.add_subplot(gs[2, 0], sharex=ax_map)
    cax = fig.add_subplot(gs[:, 1])

    fig.suptitle("ENSO evolution 1982\u2013present: Ni\u00f1o-3.4 SST anomaly and tropical Pacific response",
                 fontsize=15, fontweight="bold", y=0.975)

    # ----------------------------------------------------------- top panel
    daily_dates = df["date"]
    daily_vals = df["anomaly_c"]
    x0 = mdates.date2num(daily_dates.min())
    x1 = mdates.date2num(daily_dates.max())
    ymin, ymax = daily_vals.min() - 0.5, daily_vals.max() + 0.6

    daily_index = pd.date_range(dates.min(), dates.max(), freq="D")
    wash = series.reindex(daily_index).interpolate(method="time").ffill().bfill()
    wash_arr = wash.to_numpy()
    wash_x = mdates.date2num(daily_index)
    im = ax_top.imshow(np.ma.masked_all((1, len(wash_arr))), aspect="auto", cmap=CMAP, norm=NORM,
                       extent=[mdates.date2num(daily_index[0]), mdates.date2num(daily_index[-1]), ymin, ymax],
                       alpha=0.45, zorder=0, interpolation="bilinear")
    sc = ax_top.scatter([], [], s=2.5, color="0.35", alpha=0.30, zorder=2,
                        rasterized=True)
    daily_x = mdates.date2num(daily_dates)
    daily_xy = np.column_stack([daily_x, daily_vals.to_numpy()])
    sc.set_offsets(daily_xy[:0])   # revealed progressively in update()

    ax_top.axhline(0, color="black", linewidth=0.7, alpha=0.6)
    ax_top.axhline(WARM_THRESHOLD, color="gray", linewidth=0.7, linestyle="--")
    ax_top.axhline(COLD_THRESHOLD, color="gray", linewidth=0.7, linestyle="--")
    ax_top.set_xlim(x0, x1)
    ax_top.set_ylim(ymin, ymax)
    ax_top.xaxis.set_major_locator(mdates.YearLocator(2))
    ax_top.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_top.tick_params(axis="x", labelsize=8)
    ax_top.set_ylabel("Anomaly (\u00b0C)")
    ax_top.grid(alpha=0.2)

    (line,) = ax_top.plot([], [], color="black", linewidth=1.8, zorder=4)
    dot = ax_top.scatter([], [], s=80, zorder=6, edgecolor="0.1", linewidth=1.2)
    cursor = ax_top.axvline(x0, color="0.2", linewidth=0.9, alpha=0.6, zorder=3)

    # peak markers for El Nino / La Nina events, revealed when reached
    peak_artists = []
    for ep in episodes:
        if ep["state"] == 0:
            continue
        xd = mdates.date2num(dates[ep["peak"]])
        yv = vals[ep["peak"]]
        up = ep["state"] == 1
        mk = ax_top.scatter([xd], [yv], marker="v" if up else "^", s=28, zorder=5,
                            color=STATE_EDGE[ep["state"]], visible=False)
        tx = ax_top.text(xd, yv + (0.16 if up else -0.16), f"'{dates[ep['peak']]:%y}",
                         ha="center", va="bottom" if up else "top", fontsize=7,
                         color=STATE_EDGE[ep["state"]], visible=False)
        peak_artists.append((ep["peak"], mk, tx))

    badge = ax_top.text(0.0, 1.04, "", transform=ax_top.transAxes, ha="left", va="bottom",
                        fontsize=12.5, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.35", facecolor="0.8", edgecolor="none"))
    handles = [
        Line2D([0], [0], marker="o", linestyle="", color="0.35", alpha=0.6, markersize=4),
        Line2D([0], [0], color="black", linewidth=1.8),
    ]
    ax_top.legend(handles, ["Daily anomaly", "3-month smoothed"], loc="lower right",
                  bbox_to_anchor=(1.0, 1.02), ncol=2, frameon=False, fontsize=8.5)

    # ---------------------------------------------------------- map panel
    ax_map.set_xlim(LON_MIN, LON_MAX)
    ax_map.set_ylim(-20, 20)
    ax_map.set_facecolor("#eef5fb")
    ax_map.axhline(0, color="#7fb3e0", linewidth=1, zorder=1)
    ax_map.set_ylabel("Latitude")
    ax_map.set_title("Schematic tropical Pacific (map view)", fontsize=10.5, loc="left", pad=4)
    ticks = [140, 160, 180, 200, 220, 240, 260, 280]
    ax_map.set_xticks(ticks)
    ax_map.tick_params(axis="x", labelbottom=False)
    ax_map.set_yticks([-15, 0, 15])
    ax_map.set_yticklabels(["15\u00b0S", "Eq", "15\u00b0N"])

    for x, w, lab in [(LON_MIN, 20, "Indonesian\nArchipelago"), (278, LON_MAX - 278, "South\nAmerica")]:
        ax_map.add_patch(Rectangle((x, -20), w, 40, color="#a6d9a1", zorder=2))
        ax_map.text(x + w / 2, 0, lab, ha="center", va="center", fontsize=8,
                    fontweight="bold", rotation=90, zorder=3)

    ax_map.add_patch(Rectangle((190, -5), 50, 10, fill=False, linestyle="--",
                               edgecolor="0.2", linewidth=1.3, zorder=4))
    ax_map.text(215, 6.3, "Ni\u00f1o-3.4 (170\u2013120\u00b0W, 5\u00b0S\u20135\u00b0N)", ha="center",
                va="bottom", fontsize=8, zorder=4)

    warm = Ellipse((185, 0), 48, 9, facecolor=CMAP(NORM(0)), edgecolor="0.5",
                   linewidth=0.8, zorder=3)
    core = Ellipse((185, 0), 24, 4.5, facecolor=CMAP(NORM(0)), edgecolor="none", zorder=3.5)
    ax_map.add_patch(warm)
    ax_map.add_patch(core)

    qx = np.arange(165, 266, 20.0)
    qy = np.concatenate([np.full(len(qx), 12.5), np.full(len(qx), -12.5)])
    qx2 = np.concatenate([qx, qx])
    quiv = ax_map.quiver(qx2, qy, -np.full(len(qx2), 7.0), np.zeros(len(qx2)),
                         angles="xy", scale_units="xy", scale=1, pivot="mid",
                         width=0.004, color="0.35", zorder=4)

    peak_label = ax_map.text(1.0, 1.03, "", transform=ax_map.transAxes, ha="right", va="bottom",
                             fontsize=10.5, fontweight="bold", zorder=7,
                             bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                                       edgecolor="0.4"))
    info = ax_map.text(0.5, 0.03, "", transform=ax_map.transAxes, ha="center", va="bottom",
                       fontsize=9, zorder=7,
                       bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                                 edgecolor="0.7", alpha=0.9))

    # ---------------------------------------------------- cross-section
    ax_sec.set_xlim(LON_MIN, LON_MAX)
    ax_sec.set_ylim(300, 0)
    ax_sec.set_xticks(ticks)
    ax_sec.set_xticklabels([lon_label(v) for v in ticks], fontsize=8)
    ax_sec.set_xlabel("Longitude")
    ax_sec.set_ylabel("Depth (m)")
    ax_sec.set_title("Equatorial cross-section: thermocline depth (schematic)",
                     fontsize=10.5, loc="left", pad=4)
    ax_sec.set_facecolor("#dbe9f6")
    for x, w in [(LON_MIN, 20), (278, LON_MAX - 278)]:
        ax_sec.add_patch(Rectangle((x, 0), w, 300, color="#a6d9a1", zorder=3))
    sec_lons = np.linspace(140, 278, 60)
    sec_fill = [None]
    (thermo_line,) = ax_sec.plot([], [], color="0.15", linewidth=2, zorder=5)
    wp_line = ax_sec.axvline(185, color="0.2", linestyle=":", linewidth=1, zorder=5)
    ax_sec.text(0.5, 0.08, "cold, deep water", transform=ax_sec.transAxes, ha="center",
                va="bottom", fontsize=8, color="#2b5c8a", zorder=6)

    # ------------------------------------------------------------ colorbar
    sm = ScalarMappable(norm=NORM, cmap=CMAP)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("3-month smoothed anomaly (\u00b0C):  La Ni\u00f1a (blue) \u2194 El Ni\u00f1o (red)",
                   fontsize=8.5)
    cbar.ax.tick_params(labelsize=8)

    # -------------------------------------------------------------- update
    def update(i):
        a = vals[i]
        rgba = CMAP(NORM(a))
        p = pacific_state(a)
        ep = peak_at.get(i)
        st = int(state[i])

        # top panel
        xi = mdates.date2num(dates[i])
        k = int(np.searchsorted(wash_x, xi, side="right"))
        im.set_data(np.ma.masked_array(wash_arr[None, :], mask=(np.arange(len(wash_arr)) >= k)[None, :]))
        sc.set_offsets(daily_xy[: int(np.searchsorted(daily_x, xi, side="right"))])
        line.set_data(mdates.date2num(dates[: i + 1]), vals[: i + 1])
        dot.set_offsets([[xi, a]])
        dot.set_facecolor(rgba)
        cursor.set_xdata([xi, xi])
        badge.set_text(f"{STATE_NAME[st]}   {dates[i]:%b %Y}   {a:+.2f} \u00b0C"
                       + ("   \u2014 PEAK" if ep else ""))
        badge.get_bbox_patch().set_facecolor(rgba)
        badge.set_color(text_color_for(rgba))
        for idx, mk, tx in peak_artists:
            vis = i >= idx
            mk.set_visible(vis)
            tx.set_visible(vis)

        # map
        warm.set_center((p["lon"], 0))
        warm.set_width(2 * p["halfwidth"])
        warm.set_facecolor((*rgba[:3], 0.85))
        core.set_center((p["lon"], 0))
        core.set_width(p["halfwidth"])
        core.set_facecolor(CMAP(NORM(np.clip(1.6 * a, -COLOR_CLIP, COLOR_CLIP))))
        if ep:
            warm.set_edgecolor(STATE_EDGE[ep["state"]])
            warm.set_linewidth(3.5)
            peak_label.set_text(f"{STATE_NAME[ep['state']]} PEAK  \u00b7  "
                                f"{dates[ep['peak']]:%b %Y}  \u00b7  {vals[ep['peak']]:+.2f} \u00b0C")
            peak_label.set_visible(True)
        else:
            warm.set_edgecolor("0.5")
            warm.set_linewidth(0.8)
            peak_label.set_visible(False)
        quiv.set_UVC(-np.full(len(qx2), 7.0 * p["trade"]), np.zeros(len(qx2)))
        info.set_text(f"Warm-pool centre {lon_label(p['lon'])}   |   trade winds "
                      f"{p['trade'] * 100:.0f}% of normal   |   thermocline tilt "
                      f"{2 * p['slope']:.0f} m (W\u2013E)")

        # cross-section
        f = (sec_lons - 140) / 138.0
        z = 100 + p["slope"] * (1 - 2 * f)
        if sec_fill[0] is not None:
            sec_fill[0].remove()
        sec_fill[0] = ax_sec.fill_between(sec_lons, 0, z, color=rgba, alpha=0.85, zorder=2)
        thermo_line.set_data(sec_lons, z)
        wp_line.set_xdata([p["lon"], p["lon"]])

    frames = list(range(0, n, step))
    if frames[-1] != n - 1:
        frames.append(n - 1)
    return fig, update, frames, dict(series=series, state=state, episodes=episodes)


# ==================================================================== main
def ensure_ffmpeg():
    """True if an ffmpeg binary is usable (system one, or pip's imageio-ffmpeg)."""
    if animation.writers.is_available("ffmpeg"):
        return True
    try:
        import imageio_ffmpeg
        matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        return animation.writers.is_available("ffmpeg")
    except Exception:
        return False


def render(df, out, step, fps, dpi, verbose=False):
    """Render one animation file (.mp4 or .gif)."""
    fig, update, frames, info = build(df, step=step)
    if verbose:
        for ep in info["episodes"]:
            d = info["series"].index[ep["peak"]]
            print(f"{STATE_NAME[ep['state']]:13s} peak {d:%b %Y}  "
                  f"{info['series'].iloc[ep['peak']]:+.2f} C")
    anim = animation.FuncAnimation(fig, update, frames=frames, blit=False)
    if out.lower().endswith(".mp4"):
        # yuv420p + even dimensions = plays in PowerPoint, Keynote, QuickTime
        writer = animation.FFMpegWriter(
            fps=fps, bitrate=4000, codec="libx264",
            extra_args=["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2:color=white",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    else:
        writer = animation.PillowWriter(fps=fps)
    anim.save(out, writer=writer, dpi=dpi,
              progress_callback=lambda i, n: print(f"\r{out}: frame {i + 1}/{n}", end="")
              if i % 25 == 0 else None)
    plt.close(fig)
    print(f"\nSaved {out}")


def main():
    ap = argparse.ArgumentParser(description="Saves BOTH an .mp4 and a .gif by default.")
    ap.add_argument("csv", nargs="?", default=CSV_PATH)
    ap.add_argument("out", nargs="?", default=OUT_PATH,
                    help="output base name; extension is ignored, both formats are written")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--mp4-step", type=int, default=1, help="months per MP4 frame")
    ap.add_argument("--gif-step", type=int, default=2, help="months per GIF frame")
    ap.add_argument("--mp4-dpi", type=int, default=110)
    ap.add_argument("--gif-dpi", type=int, default=70)
    ap.add_argument("--only", choices=["mp4", "gif"], help="write just one format")
    args = ap.parse_args()

    base = args.out.rsplit(".", 1)[0] if args.out.lower().endswith((".gif", ".mp4")) else args.out
    df = load_data(args.csv)

    if args.only != "gif":
        if ensure_ffmpeg():
            render(df, base + ".mp4", args.mp4_step, args.fps, args.mp4_dpi, verbose=True)
        else:
            print("ffmpeg not found - skipping MP4. Fix with either:\n"
                  "  brew install ffmpeg\n"
                  "  /usr/bin/python3 -m pip install imageio-ffmpeg")
    if args.only != "mp4":
        render(df, base + ".gif", args.gif_step, args.fps, args.gif_dpi,
               verbose=(args.only == "gif"))


if __name__ == "__main__":
    main()