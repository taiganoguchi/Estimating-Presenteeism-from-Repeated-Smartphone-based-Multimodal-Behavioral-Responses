"""Cohort key construction (per video) — ported from cell 26."""
from __future__ import annotations
import numpy as np
import pandas as pd


def _bin_with_edges(v, edges, labels):
    try:
        v = float(v)
        if not np.isfinite(v):
            return labels[-1]
        idx = int(np.digitize([v], edges, right=True)[0])
        idx = max(0, min(idx, len(labels) - 2))
        return labels[idx]
    except Exception:
        return labels[-1]


def _bin_age(v, age_cfg):
    if age_cfg.get("mode", "decade") == "decade":
        try:
            v = float(v)
            if not np.isfinite(v):
                return "age:UNK"
            return f"age:{int(v // 10) * 10}s"
        except Exception:
            return "age:UNK"
    edges = age_cfg.get("edges", [0, 30, 40, 50, 60, 200])
    labels = [f"age:{edges[i]}-{edges[i+1]-1}" for i in range(len(edges) - 1)] + ["age:UNK"]
    return _bin_with_edges(v, edges, labels)


def _bin_commute(v, edges):
    labels = ["commute:<=30", "commute:31-60", "commute:61+", "commute:UNK"]
    return _bin_with_edges(v, edges, labels)


def _bin_sleep(v, edges):
    labels = ["sleep:<=5h", "sleep:6-7h", "sleep:8h+", "sleep:UNK"]
    return _bin_with_edges(v, edges, labels)


def _bin_workh(v, edges):
    labels = ["workh:<=6", "workh:7-9", "workh:10+", "workh:UNK"]
    return _bin_with_edges(v, edges, labels)


def _bin_place(x):
    if pd.isna(x):
        return "place:UNK"
    x = str(x).strip().lower()
    if any(k in x for k in ["home", "house", "自宅"]):
        return "place:home"
    if any(k in x for k in ["office", "work", "職場"]):
        return "place:office"
    if any(k in x for k in ["out", "通勤", "外出"]):
        return "place:out"
    return "place:OTH"


def _bin_state(x):
    if pd.isna(x):
        return "state:UNK"
    x = str(x).strip().lower()
    if any(k in x for k in ["pre", "出勤前"]):
        return "state:pre"
    if any(k in x for k in ["work", "勤務", "during"]):
        return "state:during"
    if any(k in x for k in ["post", "退勤後"]):
        return "state:post"
    if any(k in x for k in ["holiday", "休暇", "休日"]):
        return "state:holiday"
    return "state:OTH"


def build_cohort(static_features: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Returns df_cohort + dict of per-video lookup dicts."""
    co = cfg.get("cohort", {})
    use_parts_cfg = list(co.get("use_parts", ["sex", "age_bin", "role"]))
    plain_map = dict(co.get("plain_map", {
        "static_gender": "sex",
        "static_role": "role",
        "static_education": "edu",
        "survey_place": "place",
        "survey_state": "state",
    }))
    bins = co.get("bins", {})
    rare = co.get("rare", {"min_count": 10, "label": "COHORT_RARE"})

    df = static_features.copy()
    df["age_bin"] = df["static_age"].map(lambda v: _bin_age(v, bins.get("age", {}))) if "static_age" in df else "age:UNK"
    df["commute_bin"] = (
        df["static_commute_minutes_oneway"].map(lambda v: _bin_commute(v, bins.get("commute", {}).get("edges", [0, 30, 60, 9999])))
        if "static_commute_minutes_oneway" in df else "commute:UNK"
    )
    df["sleep_bin"] = (
        df["survey_sleep_hours"].map(lambda v: _bin_sleep(v, bins.get("sleep", {}).get("edges", [0, 5, 7, 24])))
        if "survey_sleep_hours" in df else "sleep:UNK"
    )
    df["workh_bin"] = (
        df["survey_work_hours_today"].map(lambda v: _bin_workh(v, bins.get("workh", {}).get("edges", [0, 6, 9, 24])))
        if "survey_work_hours_today" in df else "workh:UNK"
    )
    df["place_bin"] = df["survey_place"].map(_bin_place) if "survey_place" in df else "place:UNK"
    df["state_bin"] = df["survey_state"].map(_bin_state) if "survey_state" in df else "state:UNK"

    def _norm_cat(prefix, v):
        s = "UNK" if pd.isna(v) else str(v).strip()
        return f"{prefix}:{s}"

    for col, pfx in plain_map.items():
        if col in df.columns:
            df[pfx] = df[col].map(lambda v, _p=pfx: _norm_cat(_p, v))

    use_parts = [p for p in use_parts_cfg if p in df.columns]

    def _make_key(row):
        vals = [row[p] if p in row and pd.notna(row[p]) else f"{p}:UNK" for p in use_parts]
        return "|".join(vals) if vals else "UNK"

    df["cohort_key"] = df.apply(_make_key, axis=1)

    vc = df["cohort_key"].value_counts()
    rare_idx = set(vc[vc < int(rare.get("min_count", 10))].index)
    if rare_idx:
        df.loc[df["cohort_key"].isin(rare_idx), "cohort_key"] = str(rare.get("label", "COHORT_RARE"))

    def _by_video(col):
        return (
            df[["video_id", col]]
            .dropna(subset=["video_id"])
            .drop_duplicates(subset=["video_id"], keep="last")
            .set_index("video_id")[col].to_dict()
        )

    return df, {
        "cohort_by_video": _by_video("cohort_key"),
        "sleep_bin_by_video": _by_video("sleep_bin"),
        "workh_bin_by_video": _by_video("workh_bin"),
        "place_bin_by_video": _by_video("place_bin"),
        "state_bin_by_video": _by_video("state_bin"),
    }
