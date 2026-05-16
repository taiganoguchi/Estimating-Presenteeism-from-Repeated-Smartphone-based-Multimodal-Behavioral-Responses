"""Stage 3a: build static features (users + surveys CSV merge) and cohort.

Ported from pipeline.ipynb cells 14-16, 26.
"""
from __future__ import annotations
from pathlib import Path
import csv, os
import numpy as np
import pandas as pd


def _read_by_indices(path: Path, indices: dict, names_map: dict, ingest: dict) -> pd.DataFrame:
    """Read a CSV using 0-based column indices, skipping the first 2 header rows."""
    read_kwargs = dict(
        engine="python",
        encoding=ingest.get("encoding", "utf-8-sig"),
        sep=ingest.get("sep", ","),
        header=None,
        skiprows=2,
        dtype=str,
        on_bad_lines=ingest.get("on_bad_lines", "skip"),
        usecols=list(indices.values()),
    )
    quotechar = ingest.get("quotechar", '"')
    try:
        if quotechar is not None:
            return pd.read_csv(path, **read_kwargs).pipe(
                lambda df: df.set_axis(list(names_map.keys()), axis=1)
            )
        return pd.read_csv(path, **read_kwargs, quoting=csv.QUOTE_NONE, escapechar="\\").pipe(
            lambda df: df.set_axis(list(names_map.keys()), axis=1)
        )
    except Exception:
        return pd.read_csv(path, **read_kwargs, quoting=csv.QUOTE_NONE, escapechar="\\").pipe(
            lambda df: df.set_axis(list(names_map.keys()), axis=1)
        )


def _to_video_id_from_mp4(x: str):
    x = str(x)
    if not x or x.lower() == "nan":
        return np.nan
    base = os.path.basename(x)
    if base.lower().endswith(".mp4"):
        base = base[:-4]
    return base or np.nan


def build_static_features(cfg: dict) -> pd.DataFrame:
    """Read users/surveys CSVs and return merged static_features dataframe."""
    root = Path(cfg["paths"]["root"]).resolve()
    users_path = (root / cfg["paths"]["users_csv"]).resolve()
    surveys_path = (root / cfg["paths"]["surveys_csv"]).resolve()
    assert users_path.exists(), f"users CSV not found: {users_path}"
    assert surveys_path.exists(), f"surveys CSV not found: {surveys_path}"

    ing = cfg.get("ingest", {"encoding": "utf-8-sig", "sep": ",", "quotechar": '"', "on_bad_lines": "skip"})
    idxu = cfg["column_indices"]["users"]
    idxs = cfg["column_indices"]["surveys"]
    users_cols = {
        "employee_id": idxu["employee_id"],
        "age": idxu["age"],
        "gender": idxu["gender"],
        "education": idxu["education"],
        "role": idxu["role"],
        "commute_minutes_oneway": idxu["commute_minutes_oneway"],
    }
    surveys_cols = {
        "employee_id": idxs["employee_id"],
        "place": idxs["place"],
        "state": idxs["state"],
        "sleep_hours": idxs["sleep_hours"],
        "video_mp4": idxs["video_mp4"],
        "work_hours_today": idxs["work_hours_today"],
    }

    df_users = _read_by_indices(users_path, idxu, users_cols, ing)
    df_surv = _read_by_indices(surveys_path, idxs, surveys_cols, ing)

    for c in df_users.columns:
        df_users[c] = df_users[c].astype(str).str.strip()
    for c in df_surv.columns:
        df_surv[c] = df_surv[c].astype(str).str.strip()
    for c in ["age", "commute_minutes_oneway"]:
        if c in df_users:
            df_users[c] = pd.to_numeric(df_users[c], errors="coerce")
    for c in ["sleep_hours", "work_hours_today"]:
        if c in df_surv:
            df_surv[c] = pd.to_numeric(df_surv[c], errors="coerce")

    df_users["user_id"] = df_users["employee_id"].astype(str)
    df_surv["user_id"] = df_surv["employee_id"].astype(str)
    df_surv["video_id"] = df_surv["video_mp4"].map(_to_video_id_from_mp4)

    mu = df_users["user_id"].str.match(r"^S\d{5}$", na=False)
    ms = df_surv["user_id"].str.match(r"^S\d{5}$", na=False)
    df_users = df_users.loc[mu].copy()
    df_surv = df_surv.loc[ms].copy()

    cu = ["user_id", "age", "gender", "education", "role", "commute_minutes_oneway"]
    cs = ["user_id", "video_id", "place", "state", "sleep_hours", "work_hours_today"]
    u = df_users[cu].drop_duplicates(subset=["user_id"])
    s = df_surv[cs]
    static_features = s.merge(u, on="user_id", how="left")

    rename_map = {
        "age": "static_age",
        "gender": "static_gender",
        "education": "static_education",
        "role": "static_role",
        "commute_minutes_oneway": "static_commute_minutes_oneway",
        "place": "survey_place",
        "state": "survey_state",
        "sleep_hours": "survey_sleep_hours",
        "work_hours_today": "survey_work_hours_today",
    }
    static_features = static_features.rename(columns=rename_map)

    front = [
        "user_id", "video_id",
        "survey_place", "survey_state", "survey_sleep_hours", "survey_work_hours_today",
        "static_age", "static_gender", "static_education", "static_role", "static_commute_minutes_oneway",
    ]
    cols = [c for c in front if c in static_features.columns] + [
        c for c in static_features.columns if c not in front
    ]
    return static_features[cols]
