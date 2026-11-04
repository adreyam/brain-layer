#!/usr/bin/env python3
"""
Violent-crime hotspot prediction pipeline (Chicago, 250 m grid)

Based on: Alhammadi (2025), "Predicting Violent Crime Hotspots", RIT thesis.
Re-implements the thesis's empirical demonstration (KDE baseline vs Random Forest /
Gradient Boosting / XGBoost) with the leakage and consistency problems fixed.

Task
    For every grid cell and forecast period t (default 7 days), predict whether the cell
    will record >= 1 violent incident in period t, using ONLY information from periods < t.

Differences from the thesis (deliberate)
    * Features are rolling/expanding and strictly lagged. The thesis appendix lists
      static per-cell aggregates (total_crimes, avg_hour, ...) computed over the whole
      training period, which leak the target for rows inside the training window.
    * Same-period spatial lag (3x3 neighbourhood), recency and trend features are added.
    * PAI / hit-rate are computed per forecast period (top-k% of cells each week), the way a
      patrol-allocation tool would be used, and averaged over the test window.
    * Evaluation uses a strict chronological split; a gap-free design is safe because every
      feature for period t only uses periods < t.
    * Coverage is defined against ALL cells in the study area (cells with any historical
      incident), stated explicitly.

Usage
    # real data: download the Chicago "Crimes - 2001 to Present" CSV (dataset ijzp-q8t2)
    python hotspot_model.py --csv Crimes_-_2001_to_Present.csv --outdir out

    # add context layers (any CSV with latitude/longitude columns)
    python hotspot_model.py --csv crimes.csv \\
        --static-layer police=police_stations.csv --static-layer hospital=hospitals.csv \\
        --dated-layer streetlight_out=311_streetlights_out.csv --outdir out

    # smoke test without data (add --synthetic-layers to also test the layer code path)
    python hotspot_model.py --synthetic --synthetic-layers --outdir out_synth

Context layers
    --static-layer name=file.csv   fixed places (police stations, hospitals, schools, bars, transit stops).
                                   Adds per cell: distance to nearest (m) and count within ~600 m.
    --dated-layer  name=file.csv   events with a date column (311 street-light outages, vacant-building
                                   reports, new licences). Adds per cell and per 3x3 neighbourhood: counts in
                                   the previous 4 weeks / 3 months, strictly lagged like the crime features.
    Columns are auto-detected: latitude/lat, longitude/lon/lng, and for dated layers a column such as
    "Creation Date", "Date" or "Created Date". When layers are given, the script also fits a crime-only
    Gradient Boosting model so you can see whether the layers really improve precision.

Note: predicted scores are for RANKING cells (hotspot selection); they are not calibrated probabilities.

Outputs (in --outdir)
    metrics.json            all metrics for every model
    feature_importance.csv  permutation importance on the test set (best model)
    next_period_risk.csv    risk score per cell for the period after the data ends
    gain_curves.png         hit rate vs. area covered, all models
    best_model.joblib       fitted best model + metadata
"""
from __future__ import annotations

import argparse
import json
import math
import os
import warnings

import numpy as np
import pandas as pd
from scipy.ndimage import convolve, distance_transform_edt, gaussian_filter, uniform_filter
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

warnings.filterwarnings("ignore", category=FutureWarning)

VIOLENT_TYPES = {
    "HOMICIDE", "ROBBERY", "ASSAULT", "BATTERY",
    "CRIMINAL SEXUAL ASSAULT", "CRIM SEXUAL ASSAULT",
}
CELL_M = 250.0
M_PER_DEG_LAT = 111_320.0


# --------------------------------------------------------------------------- data
def load_chicago(csv_path: str, types: set[str]) -> pd.DataFrame:
    cols = ["Date", "Primary Type", "Latitude", "Longitude"]
    df = pd.read_csv(csv_path, usecols=cols)
    df = df.dropna(subset=["Latitude", "Longitude"])
    df["Primary Type"] = df["Primary Type"].str.upper().str.strip()
    df = df[df["Primary Type"].isin(types)]
    df["dt"] = pd.to_datetime(df["Date"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
    df = df.dropna(subset=["dt"])
    df = df[(df.Latitude.between(41.6, 42.1)) & (df.Longitude.between(-87.95, -87.5))]
    return df.rename(columns={"Latitude": "lat", "Longitude": "lon"})[["dt", "lat", "lon"]]


def make_synthetic(seed: int = 0, n_years: int = 6) -> pd.DataFrame:
    """Chicago-shaped synthetic incidents: concentrated hotspots + weekly/seasonal cycles
    + short-term near-repeat bursts. Only for smoke-testing the pipeline."""
    rng = np.random.default_rng(seed)
    n_centres = 120
    centres = np.column_stack([rng.uniform(41.65, 42.02, n_centres), rng.uniform(-87.80, -87.55, n_centres)])
    weights = rng.pareto(1.3, n_centres) + 0.1
    weights /= weights.sum()
    days = pd.date_range("2015-01-01", periods=365 * n_years, freq="D")
    rows = []
    for d in days:
        season = 1 + 0.25 * math.sin(2 * math.pi * (d.dayofyear - 80) / 365)
        weekend = 1.25 if d.dayofweek >= 5 else 1.0
        n = rng.poisson(110 * season * weekend)
        idx = rng.choice(n_centres, n, p=weights)
        lat = centres[idx, 0] + rng.normal(0, 0.006, n)
        lon = centres[idx, 1] + rng.normal(0, 0.008, n)
        hrs = rng.integers(0, 24, n)
        # near-repeat burst: re-sample a few recent locations
        if rows and rng.random() < 0.6:
            prev = rows[-1]
            k = min(len(prev[0]), 15)
            lat = np.r_[lat, prev[0][:k] + rng.normal(0, 0.002, k)]
            lon = np.r_[lon, prev[1][:k] + rng.normal(0, 0.002, k)]
            hrs = np.r_[hrs, rng.integers(0, 24, k)]
        rows.append((lat, lon))
        ts = d + pd.to_timedelta(hrs, unit="h")
        rows[-1] = (lat, lon, ts)
    lat = np.concatenate([r[0] for r in rows])
    lon = np.concatenate([r[1] for r in rows])
    ts = np.concatenate([r[2].values for r in rows])
    return pd.DataFrame({"dt": pd.to_datetime(ts), "lat": lat, "lon": lon})


# --------------------------------------------------------------------------- grid
def to_grid(df: pd.DataFrame, cell_m: float = CELL_M):
    lat0 = df.lat.mean()
    m_per_deg_lon = M_PER_DEG_LAT * math.cos(math.radians(lat0))
    y = (df.lat.values - df.lat.min()) * M_PER_DEG_LAT
    x = (df.lon.values - df.lon.min()) * m_per_deg_lon
    gy = (y // cell_m).astype(int)
    gx = (x // cell_m).astype(int)
    meta = dict(lat_min=float(df.lat.min()), lon_min=float(df.lon.min()),
                m_per_deg_lon=m_per_deg_lon, cell_m=cell_m)
    return gy, gx, meta


def build_count_cubes(df: pd.DataFrame, horizon: int, cell_m: float):
    """Return per-period count cubes (T,H,W): all / night / weekend incidents, + period start dates."""
    df = df.copy()
    gy, gx, meta = to_grid(df, cell_m)
    H, W = gy.max() + 1, gx.max() + 1
    t0 = df.dt.min().normalize()
    day_idx = ((df.dt.dt.normalize() - t0).dt.days).values
    p = day_idx // horizon
    T = p.max() + 1
    # drop a trailing partial period so every period is the same length
    last_full = (day_idx.max() + 1) // horizon
    keep = p < last_full
    p, gy, gx, df = p[keep], gy[keep], gx[keep], df[keep]
    T = last_full
    night = ((df.dt.dt.hour >= 20) | (df.dt.dt.hour < 6)).values
    weekend = (df.dt.dt.dayofweek >= 5).values

    def cube(mask=None):
        c = np.zeros((T, H, W), dtype=np.int16)
        sel = slice(None) if mask is None else mask
        np.add.at(c, (p[sel], gy[sel], gx[sel]), 1)
        return c

    starts = t0 + pd.to_timedelta(np.arange(T) * horizon, unit="D")
    return cube(), cube(night), cube(weekend), starts, meta



# ------------------------------------------------------------------ context layers
LAT_KEYS = ("latitude", "lat")
LON_KEYS = ("longitude", "lon", "lng", "long")
DATE_KEYS = ("creation date", "created date", "created_date", "creation_date", "date", "date issued",
             "issue date", "start date", "start_date", "datetime", "timestamp")


def _split_spec(spec: str):
    if "=" not in spec:
        raise SystemExit(f"layer spec must look like name=path.csv, got: {spec}")
    name, path = spec.split("=", 1)
    return name.strip().replace(" ", "_"), path.strip().strip('"')


def load_layer(path: str, need_date: bool) -> pd.DataFrame:
    df = pd.read_csv(path)
    low = {c.lower().strip(): c for c in df.columns}
    lat = next((low[k] for k in LAT_KEYS if k in low), None)
    lon = next((low[k] for k in LON_KEYS if k in low), None)
    if lat is None or lon is None:
        raise SystemExit(f"{path}: need latitude/longitude columns, found {list(df.columns)[:15]}")
    out = pd.DataFrame({"lat": pd.to_numeric(df[lat], errors="coerce"),
                        "lon": pd.to_numeric(df[lon], errors="coerce")})
    if need_date:
        dcol = next((low[k] for k in DATE_KEYS if k in low), None)
        if dcol is None:
            raise SystemExit(f"{path}: dated layer needs a date column, found {list(df.columns)[:15]}")
        try:
            out["dt"] = pd.to_datetime(df[dcol], format="mixed", errors="coerce")
        except Exception:
            out["dt"] = pd.to_datetime(df[dcol], errors="coerce")
    return out.dropna().reset_index(drop=True)


def points_to_cells(lat, lon, meta, H, W):
    gy = np.floor((np.asarray(lat) - meta["lat_min"]) * M_PER_DEG_LAT / meta["cell_m"]).astype(int)
    gx = np.floor((np.asarray(lon) - meta["lon_min"]) * meta["m_per_deg_lon"] / meta["cell_m"]).astype(int)
    ok = (gy >= 0) & (gy < H) & (gx >= 0) & (gx < W)
    return gy, gx, ok


def build_layers(static_specs, dated_specs, meta, H, W, T, starts, horizon, cell_m):
    layers = []
    for spec in static_specs:
        name, path = _split_spec(spec)
        d = load_layer(path, need_date=False)
        gy, gx, ok = points_to_cells(d.lat.values, d.lon.values, meta, H, W)
        if ok.sum() == 0:
            print(f"[warn] layer '{name}': no points fall inside the crime grid, skipped")
            continue
        cnt = np.zeros((H, W), np.float32)
        np.add.at(cnt, (gy[ok], gx[ok]), 1)
        layers.append(dict(name=name, kind="static", count=cnt, cell_m=cell_m))
        print(f"layer '{name}' (static): {int(ok.sum()):,} points in grid")
    for spec in dated_specs:
        name, path = _split_spec(spec)
        d = load_layer(path, need_date=True)
        gy, gx, ok = points_to_cells(d.lat.values, d.lon.values, meta, H, W)
        p = ((d.dt.dt.normalize() - starts[0]).dt.days.values // horizon)
        ok = ok & (p >= 0) & (p < T)
        if ok.sum() == 0:
            print(f"[warn] layer '{name}': no events inside the grid/time range, skipped")
            continue
        cube = np.zeros((T, H, W), np.int16)
        np.add.at(cube, (p[ok], gy[ok], gx[ok]), 1)
        layers.append(dict(name=name, kind="dated", cube=cube))
        print(f"layer '{name}' (dated): {int(ok.sum()):,} events in grid/time range")
    return layers


def extend_layers(layers):
    """Append one empty future period to dated layers (used for the next-period forecast)."""
    out = []
    for L in layers:
        if L["kind"] == "dated":
            c = L["cube"]
            L = {**L, "cube": np.concatenate([c, np.zeros((1,) + c.shape[1:], c.dtype)])}
        out.append(L)
    return out


def make_synthetic_layers(df: pd.DataFrame, outdir: str, seed: int):
    """Random layers, ONLY to exercise the layer code path (they are unrelated to the synthetic crime)."""
    rng = np.random.default_rng(seed + 1)
    la, lb = df.lat.quantile([0.02, 0.98])
    oa, ob = df.lon.quantile([0.02, 0.98])
    for name, n in (("police", 22), ("hospital", 30)):
        pd.DataFrame({"Latitude": rng.uniform(la, lb, n), "Longitude": rng.uniform(oa, ob, n)}).to_csv(
            os.path.join(outdir, f"{name}.csv"), index=False)
    n = 30_000
    span = (df.dt.max() - df.dt.min()).days
    when = df.dt.min().normalize() + pd.to_timedelta(rng.integers(0, span, n), unit="D") \
        + pd.to_timedelta(rng.integers(0, 24, n), unit="h")
    pd.DataFrame({"Latitude": rng.uniform(la, lb, n), "Longitude": rng.uniform(oa, ob, n),
                  "Creation Date": when.strftime("%m/%d/%Y %I:%M:%S %p")}).to_csv(
        os.path.join(outdir, "streetlight_out.csv"), index=False)

# ------------------------------------------------------------------------ features
def lagged_sum(cube: np.ndarray, w: int) -> np.ndarray:
    """out[t] = sum of cube[t-w .. t-1] (0 where unavailable). Never includes period t."""
    cs = np.cumsum(cube, axis=0, dtype=np.float32)
    cs = np.concatenate([np.zeros((1,) + cube.shape[1:], np.float32), cs], axis=0)  # cs[k]=sum[:k]
    T = cube.shape[0]
    t = np.arange(T)
    hi = cs[t]                       # sum of [0..t-1]
    lo = cs[np.maximum(t - w, 0)]    # sum of [0..t-w-1]
    return hi - lo


def periods_since_last(cube: np.ndarray, cap: int) -> np.ndarray:
    """Periods since the last period with >=1 incident, strictly before t (cap if never)."""
    T = cube.shape[0]
    out = np.full(cube.shape, cap, dtype=np.float32)
    last = np.full(cube.shape[1:], -10**6, dtype=np.int64)
    for t in range(T):
        gap = t - last
        out[t] = np.minimum(np.where(last < 0, cap, gap), cap)
        last = np.where(cube[t] > 0, t, last)
    return out


def neighbour_sum(arr: np.ndarray) -> np.ndarray:
    k = np.ones((1, 3, 3), np.float32)
    k[0, 1, 1] = 0.0
    return convolve(arr, k, mode="constant", cval=0.0)


def build_features(cube, night, weekend, starts, horizon, layers=None):
    T, H, W = cube.shape
    f = {}
    wins_days = [7, 28, 91, 182, 364]
    wins = sorted({max(1, int(round(d / horizon))) for d in wins_days})
    cnt = cube.astype(np.float32)
    for w in wins:
        lag = lagged_sum(cnt, w)
        f[f"cnt_last_{w}p"] = lag
        f[f"occ_last_{w}p"] = lagged_sum((cnt > 0).astype(np.float32), w)
    for w in [wins[0], wins[min(2, len(wins) - 1)]]:
        f[f"nbr_last_{w}p"] = neighbour_sum(lagged_sum(cnt, w))
    # long-run rate (expanding mean) and shrunk behavioural shares
    cum = lagged_sum(cnt, T)
    n_prev = np.arange(T, dtype=np.float32).reshape(T, 1, 1)
    f["rate_expanding"] = cum / np.maximum(n_prev, 1)
    f["occ_rate_expanding"] = lagged_sum((cnt > 0).astype(np.float32), T) / np.maximum(n_prev, 1)
    for name, c in (("night", night), ("weekend", weekend)):
        num = lagged_sum(c.astype(np.float32), T)
        city = num.sum(axis=(1, 2), keepdims=True) / np.maximum(cum.sum(axis=(1, 2), keepdims=True), 1)
        f[f"{name}_share"] = (num + 5 * city) / (cum + 5)  # shrinkage prior of 5 incidents
    f["since_last"] = periods_since_last(cube, cap=max(wins) * 2)
    # trend: recent vs longer window (ratio), and citywide level
    short, longw = f[f"cnt_last_{wins[1] if len(wins) > 1 else wins[0]}p"], f[f"cnt_last_{wins[-1]}p"]
    scale = (wins[-1] / (wins[1] if len(wins) > 1 else wins[0]))
    f["trend_ratio"] = (short + 0.5) / (longw / scale + 0.5)
    city_total = cnt.sum(axis=(1, 2))
    city_lag = lagged_sum(city_total.reshape(T, 1, 1), wins[min(1, len(wins) - 1)]).reshape(T)
    f["city_level"] = np.broadcast_to(city_lag.reshape(T, 1, 1), (T, H, W)).copy()
    # smooth historical density (KDE-like surface of last ~year), also used as a feature
    dens = gaussian_filter(f[f"cnt_last_{wins[-1]}p"], sigma=(0, 1.5, 1.5))
    f["kde_year"] = dens.astype(np.float32)
    # calendar of forecast period
    mid = starts + pd.to_timedelta(horizon / 2, unit="D")
    doy = mid.dayofyear.values.astype(np.float32)
    cal = {
        "season_sin": np.sin(2 * np.pi * doy / 365.25),
        "season_cos": np.cos(2 * np.pi * doy / 365.25),
        "month": mid.month.values.astype(np.float32),
    }
    if horizon < 7:
        cal["dow"] = starts.dayofweek.values.astype(np.float32)
        cal["is_weekend"] = (starts.dayofweek.values >= 5).astype(np.float32)
    for k, v in cal.items():
        f[k] = np.broadcast_to(v.reshape(T, 1, 1), (T, H, W)).astype(np.float32)
    # ---- context layers (feature names start with "L_")
    for L in (layers or []):
        nm = L["name"]
        if L["kind"] == "static":
            cnt_l = L["count"]
            dist = np.minimum(distance_transform_edt(cnt_l == 0) * L["cell_m"], 5000.0).astype(np.float32)
            near = (uniform_filter(cnt_l, size=5, mode="constant") * 25).astype(np.float32)  # ~ +-600 m box
            f[f"L_{nm}_dist_m"] = np.broadcast_to(dist, (T, H, W))
            f[f"L_{nm}_count_nearby"] = np.broadcast_to(near, (T, H, W))
        else:
            lw = sorted({wins[min(1, len(wins) - 1)], wins[min(2, len(wins) - 1)]})
            for w in lw:
                lag = lagged_sum(L["cube"].astype(np.float32), w)  # strictly before period t
                f[f"L_{nm}_cell_last_{w}p"] = lag
                f[f"L_{nm}_nbr_last_{w}p"] = neighbour_sum(lag)
    return f, wins


# -------------------------------------------------------------------- evaluation
def hit_pai_by_period(score, y_count, cell_idx, period_idx, covs, n_cells):
    """Mean (over forecast periods) hit-rate and PAI when the top-k% of ALL study cells are patrolled."""
    out = {c: {"hit": [], "pai": []} for c in covs}
    for p in np.unique(period_idx):
        m = period_idx == p
        s, yc = score[m], y_count[m]
        total = yc.sum()
        if total == 0:
            continue
        order = np.argsort(-s, kind="stable")
        csum = np.cumsum(yc[order])
        for c in covs:
            k = max(1, int(round(c * n_cells)))
            hit = csum[min(k, len(csum)) - 1] / total
            out[c]["hit"].append(hit)
            out[c]["pai"].append(hit / c)
    return {f"{int(c * 100)}%": {"hit_rate": float(np.mean(v["hit"])), "PAI": float(np.mean(v["pai"]))}
            for c, v in out.items()}


def gain_curve(score, y_count, period_idx, n_cells, grid=np.linspace(0.01, 0.5, 50)):
    hits = []
    for c in grid:
        h = []
        for p in np.unique(period_idx):
            m = period_idx == p
            yc = y_count[m]
            if yc.sum() == 0:
                continue
            order = np.argsort(-score[m], kind="stable")
            k = max(1, int(round(c * n_cells)))
            h.append(yc[order][:k].sum() / yc.sum())
        hits.append(np.mean(h))
    return grid, np.array(hits)


def prob_metrics(y, s):
    return {"AUC": float(roc_auc_score(y, s)), "PR_AUC": float(average_precision_score(y, s)),
            "positive_rate": float(np.mean(y))}


# -------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", help="Chicago crimes CSV (portal dataset ijzp-q8t2)")
    ap.add_argument("--synthetic", action="store_true", help="use synthetic data (smoke test only)")
    ap.add_argument("--types", default=",".join(sorted(VIOLENT_TYPES)), help="comma-separated Primary Types")
    ap.add_argument("--horizon", type=int, default=7, help="forecast period length in days (1 = daily like the thesis)")
    ap.add_argument("--test-days", type=int, default=90)
    ap.add_argument("--cell-m", type=float, default=CELL_M)
    ap.add_argument("--max-train-rows", type=int, default=1_000_000, help="subsample training rows (negatives only)")
    ap.add_argument("--static-layer", action="append", default=[], metavar="NAME=CSV",
                    help="fixed places, e.g. police=stations.csv (repeatable)")
    ap.add_argument("--dated-layer", action="append", default=[], metavar="NAME=CSV",
                    help="dated events, e.g. streetlight_out=311.csv (repeatable)")
    ap.add_argument("--synthetic-layers", action="store_true", help="with --synthetic: add random test layers")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="hotspot_out")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # ---- data
    if args.synthetic:
        df = make_synthetic(args.seed)
        print("[!] SYNTHETIC data: numbers below only prove the pipeline runs, they say nothing about Chicago.")
    elif args.csv:
        df = load_chicago(args.csv, {t.strip().upper() for t in args.types.split(",")})
    else:
        ap.error("provide --csv or --synthetic")
    print(f"incidents: {len(df):,}  {df.dt.min().date()} -> {df.dt.max().date()}")

    cube, night, weekend, starts, meta = build_count_cubes(df, args.horizon, args.cell_m)
    T, H, W = cube.shape
    n_test = max(1, int(round(args.test_days / args.horizon)))
    t_cut = T - n_test
    print(f"grid {H}x{W}  periods={T} (every {args.horizon}d)  train periods<{t_cut}  test periods={n_test}")

    # study area = cells with any incident in the TRAIN window (no peeking at test)
    active = cube[:t_cut].sum(axis=0) > 0
    n_cells = int(active.sum())
    print(f"active study-area cells: {n_cells:,}")

    if args.synthetic and args.synthetic_layers:
        make_synthetic_layers(df, args.outdir, args.seed)
        args.static_layer = args.static_layer + [f"police={os.path.join(args.outdir, 'police.csv')}",
                                                 f"hospital={os.path.join(args.outdir, 'hospital.csv')}"]
        args.dated_layer = args.dated_layer + [f"streetlight_out={os.path.join(args.outdir, 'streetlight_out.csv')}"]
    layers = build_layers(args.static_layer, args.dated_layer, meta, H, W, T, starts, args.horizon, args.cell_m)

    feats, wins = build_features(cube, night, weekend, starts, args.horizon, layers)
    names = list(feats.keys())
    layer_cols = [i for i, n in enumerate(names) if n.startswith("L_")]
    crime_cols = [i for i, n in enumerate(names) if not n.startswith("L_")]
    if layer_cols:
        print(f"{len(layer_cols)} layer features added: {[names[i] for i in layer_cols]}")
    min_hist = max(wins)  # skip early periods that lack a full history window
    t_idx = np.arange(T)

    def flatten(t_range):
        t_sel = t_idx[t_range]
        X = np.stack([feats[n][t_sel][:, active] for n in names], axis=-1)  # (t, cells, F)
        y_cnt = cube[t_sel][:, active].astype(np.int32)
        p_ix = np.repeat(t_sel, n_cells)
        c_ix = np.tile(np.arange(n_cells), len(t_sel))
        return X.reshape(-1, len(names)), y_cnt.reshape(-1), p_ix, c_ix

    Xtr, ytr_c, ptr, _ = flatten(slice(min_hist, t_cut))
    Xte, yte_c, pte, cte = flatten(slice(t_cut, T))
    ytr, yte = (ytr_c > 0).astype(int), (yte_c > 0).astype(int)
    print(f"train rows {len(ytr):,} (pos {ytr.mean():.3%})  test rows {len(yte):,} (pos {yte.mean():.3%})")

    # subsample negatives for speed; keep ALL positives; reweight so probabilities stay calibrated-ish
    w_tr = np.ones(len(ytr), dtype=np.float32)
    if len(ytr) > args.max_train_rows:
        pos = np.flatnonzero(ytr == 1)
        neg = np.flatnonzero(ytr == 0)
        n_neg = max(args.max_train_rows - len(pos), len(pos))  # never fewer negatives than positives
        neg_s = rng.choice(neg, size=min(n_neg, len(neg)), replace=False)
        keep = np.sort(np.r_[pos, neg_s])
        w_tr = np.where(ytr[keep] == 0, len(neg) / len(neg_s), 1.0).astype(np.float32)
        Xtr, ytr = Xtr[keep], ytr[keep]
        print(f"subsampled train to {len(ytr):,} rows (negatives weighted x{len(neg) / len(neg_s):.1f})")

    # ---- baselines (scores for test rows)
    ix = {n: i for i, n in enumerate(names)}
    baselines = {
        "baseline_recent_count(last 4 wks)": Xte[:, ix[f"cnt_last_{wins[1] if len(wins) > 1 else wins[0]}p"]],
        "baseline_KDE(trailing year, gaussian)": Xte[:, ix["kde_year"]],
        "baseline_long_run_rate": Xte[:, ix["rate_expanding"]],
    }

    # ---- models
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=150, max_depth=12, min_samples_leaf=20, n_jobs=-1,
                                               class_weight="balanced_subsample", random_state=args.seed),
        "GradientBoosting": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_depth=6,
                                                           l2_regularization=1.0, early_stopping=True,
                                                           validation_fraction=0.1, random_state=args.seed),
    }
    try:
        from xgboost import XGBClassifier
        spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
        models["XGBoost"] = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.06, subsample=0.8,
                                          colsample_bytree=0.8, scale_pos_weight=spw, eval_metric="logloss",
                                          tree_method="hist", n_jobs=-1, random_state=args.seed)
    except ImportError:
        print("xgboost not installed -> skipping XGBoost (pip install xgboost)")

    covs = [0.05, 0.10, 0.15, 0.20, 0.25]
    results, scores = {}, {}
    for name, s in baselines.items():
        scores[name] = s
        results[name] = {**prob_metrics(yte, s), "by_coverage": hit_pai_by_period(s, yte_c, cte, pte, covs, n_cells)}
    fitted = {}
    for name, mdl in models.items():
        print(f"fitting {name} ...")
        if name == "XGBoost":
            mdl.fit(Xtr, ytr)  # handles imbalance via scale_pos_weight
        else:
            mdl.fit(Xtr, ytr, sample_weight=w_tr)  # undo negative subsampling
        s = mdl.predict_proba(Xte)[:, 1]
        scores[name], fitted[name] = s, mdl
        res = prob_metrics(yte, s)
        res["Brier"] = float(brier_score_loss(yte, s))
        res["by_coverage"] = hit_pai_by_period(s, yte_c, cte, pte, covs, n_cells)
        results[name] = res

    # ---- ablation: same GB model WITHOUT the context layers
    if layer_cols:
        print("fitting GradientBoosting_crime_only (ablation) ...")
        ab = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_depth=6, l2_regularization=1.0,
                                            early_stopping=True, validation_fraction=0.1, random_state=args.seed)
        ab.fit(Xtr[:, crime_cols], ytr, sample_weight=w_tr)
        s_ab = ab.predict_proba(Xte[:, crime_cols])[:, 1]
        scores["GradientBoosting_crime_only"] = s_ab
        r_ab = prob_metrics(yte, s_ab)
        r_ab["Brier"] = float(brier_score_loss(yte, s_ab))
        r_ab["by_coverage"] = hit_pai_by_period(s_ab, yte_c, cte, pte, covs, n_cells)
        results["GradientBoosting_crime_only"] = r_ab

    # ---- report
    print("\n=== TEST RESULTS (top-k% of study-area cells patrolled each period, averaged over test periods) ===")
    print(f"{'model':40s} {'AUC':>6s} {'PR-AUC':>7s} | PAI@5%  PAI@10% PAI@20% | hit@5%  hit@10% hit@20%")
    for name, r in results.items():
        b = r["by_coverage"]
        print(f"{name:40s} {r['AUC']:6.3f} {r['PR_AUC']:7.3f} | {b['5%']['PAI']:6.2f}  {b['10%']['PAI']:6.2f}  "
              f"{b['20%']['PAI']:6.2f} | {b['5%']['hit_rate']:6.3f}  {b['10%']['hit_rate']:6.3f}  {b['20%']['hit_rate']:6.3f}")

    best = max(fitted, key=lambda n: results[n]["by_coverage"]["10%"]["PAI"])
    print(f"\nbest ML model by PAI@10%: {best}")

    meta_out = dict(args=vars(args), grid=dict(H=int(H), W=int(W)), n_active_cells=n_cells,
                    n_periods=int(T), test_periods=int(n_test), features=names, best_model=best,
                    test_start=str(starts[t_cut].date()), test_end=str((starts[-1] + pd.Timedelta(days=args.horizon - 1)).date()),
                    synthetic=bool(args.synthetic))
    with open(os.path.join(args.outdir, "metrics.json"), "w") as fh:
        json.dump({"meta": meta_out, "results": results}, fh, indent=2)

    # ---- permutation importance (best model, subsample for speed)
    sub = rng.choice(len(yte), size=min(60_000, len(yte)), replace=False)
    pi = permutation_importance(fitted[best], Xte[sub], yte[sub], scoring="roc_auc", n_repeats=3,
                                random_state=args.seed, n_jobs=1)
    imp = pd.DataFrame({"feature": names, "auc_drop": pi.importances_mean}).sort_values("auc_drop", ascending=False)
    imp.to_csv(os.path.join(args.outdir, "feature_importance.csv"), index=False)
    print("\ntop features (permutation, test AUC drop):")
    print(imp.head(8).to_string(index=False))
    if layer_cols:
        print("\ncontext-layer features (permutation, test AUC drop):")
        print(imp[imp.feature.str.startswith("L_")].to_string(index=False))

    # ---- gain curves
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        for name, s in scores.items():
            g, h = gain_curve(s, yte_c, pte, n_cells)
            ax.plot(g * 100, h * 100, label=name, ls="--" if name.startswith("baseline") else "-")
        ax.plot([0, 50], [0, 50], color="grey", lw=0.8, label="random")
        ax.set_xlabel("% of study-area cells patrolled each period")
        ax.set_ylabel("% of next-period violent incidents captured")
        ax.set_title("Hit rate vs. area covered (temporal hold-out)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, "gain_curves.png"), dpi=150)
    except Exception as e:  # plotting is optional
        print("plot skipped:", e)

    # ---- forecast for the period after the data ends
    t_next = T
    # build features for a virtual period T by appending an empty period
    cube2 = np.concatenate([cube, np.zeros((1, H, W), np.int16)])
    night2 = np.concatenate([night, np.zeros((1, H, W), np.int16)])
    weekend2 = np.concatenate([weekend, np.zeros((1, H, W), np.int16)])
    starts2 = starts.append(pd.DatetimeIndex([starts[-1] + pd.Timedelta(days=args.horizon)]))
    f2, _ = build_features(cube2, night2, weekend2, starts2, args.horizon, extend_layers(layers))
    Xnext = np.stack([f2[n][t_next][active] for n in names], axis=-1)
    risk = fitted[best].predict_proba(Xnext)[:, 1]
    gy_a, gx_a = np.nonzero(active)
    lat_c = meta["lat_min"] + (gy_a + 0.5) * meta["cell_m"] / M_PER_DEG_LAT
    lon_c = meta["lon_min"] + (gx_a + 0.5) * meta["cell_m"] / meta["m_per_deg_lon"]
    out = pd.DataFrame({"grid_y": gy_a, "grid_x": gx_a, "lat": lat_c, "lon": lon_c, "risk": risk})
    out = out.sort_values("risk", ascending=False)
    out["risk_rank_pct"] = np.arange(1, len(out) + 1) / len(out) * 100
    out["top10pct_hotspot"] = out.risk_rank_pct <= 10
    out.to_csv(os.path.join(args.outdir, "next_period_risk.csv"), index=False)

    import joblib
    joblib.dump({"model": fitted[best], "features": names, "grid_meta": meta, "horizon": args.horizon},
                os.path.join(args.outdir, "best_model.joblib"))
    print(f"\nwrote outputs to {args.outdir}/")
    if args.synthetic:
        print("REMINDER: synthetic data - do not quote these metrics.")


if __name__ == "__main__":
    main()