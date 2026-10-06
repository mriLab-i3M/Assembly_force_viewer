"""Pure table helpers (no Qt) shared by the viewer GUI: Lid tables, filtering, pivoting,
max/min highlighting and the Worst-case search over custom Lid studies."""
import os
import re

import numpy as np
import pandas as pd

from custom_lids import CUSTOM_TABLE_NAME

COMPONENT_COLUMNS = {
    "fx": "Fx_sum",
    "fy": "Fy_sum",
    "fz": "Fz_sum",
    "normf": "normF",
    "tx": "Tx_sum",
    "ty": "Ty_sum",
    "tz": "Tz_sum",
    "normt": "normT",
}
NORM_COLUMNS = ("normF", "normT")
LID_VALUE_COLUMNS = ["Fx_sum", "Fy_sum", "Fz_sum", "normF", "Tx_sum", "Ty_sum", "Tz_sum", "normT"]
LID_TABLE_COLUMNS = ["Ring", "Lid", "N_cubes"] + LID_VALUE_COLUMNS


def component_column(name):
    """'Fx', 'Fx_Lids', 'NormF' or 'normF' -> table column ('Fx_sum', ..., 'normF')."""
    return COMPONENT_COLUMNS[str(name).replace("_Lids", "").lower()]


def is_norm_column(name):
    return str(name).lower().startswith("norm")


def lid_sort_key(name):
    """Numbered lids ('Lid 2' < 'Lid 10') first by number; anything else (Lid+Y, ...) alphabetically."""
    match = re.fullmatch(r"Lid\s*(\d+)", str(name))
    return (0, int(match.group(1)), "") if match else (1, 0, str(name))


def sorted_lids(names):
    return sorted({str(n) for n in names}, key=lid_sort_key)


def sorted_rings(values):
    numeric = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    return sorted({int(v) for v in numeric})


def parse_lid_table(df):
    """Split 'Ring_Lid' (e.g. 'Ring3_Lid 2' or 'Ring3_Lid+Y') into Ring and Lid columns.

    Tables without ring prefix (old custom studies, rows like 'Lid1') get an empty Ring.
    """
    parts = df["Ring_Lid"].astype(str).str.extract(r"^Ring(\d+)_(.*)$")
    out = df.drop(columns=["Ring_Lid"]).copy()
    if parts[0].notna().all():
        rings = parts[0].astype(int)
        lids = parts[1]
    else:
        rings = pd.Series([pd.NA] * len(df), index=df.index)
        lids = df["Ring_Lid"].astype(str)
    out.insert(0, "Ring", rings.astype("Int64"))
    out.insert(1, "Lid", lids)
    return out


def read_lid_table(path):
    return parse_lid_table(pd.read_csv(path, sep="\t"))


def filter_lid_rows(df, rings=None, lids=None):
    """Keep rows whose Ring/Lid are in the given selections (None = no filter on that axis)."""
    mask = pd.Series(True, index=df.index)
    if rings is not None and "Ring" in df:
        mask &= df["Ring"].isin(list(rings))
    if lids is not None and "Lid" in df:
        mask &= df["Lid"].isin(list(lids))
    return df[mask]


def build_lid_pivot(df, component):
    """Rings as rows, one column per lid, for one component."""
    pivot = df.pivot(index="Ring", columns="Lid", values=component_column(component))
    pivot = pivot.astype(float)
    return pivot[sorted(pivot.columns, key=lid_sort_key)]


def extreme_cells(df, columns, max_only=()):
    """Return {(row_position, column): 'max' | 'min'} for each numeric column.

    Columns listed in max_only (or named norm*) only get their maximum marked.
    """
    marks = {}
    for column in columns:
        values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
        if not np.any(~np.isnan(values)):
            continue
        high = np.nanmax(values)
        low = np.nanmin(values)
        only_max = column in max_only or is_norm_column(column)
        for row, value in enumerate(values):
            if np.isnan(value):
                continue
            if value == high:
                marks[(row, column)] = "max"
            elif value == low and not only_max and low != high:
                marks[(row, column)] = "min"
    return marks


def find_worst_lid_case(base_folder, step_names, study_name, component, lid=None):
    """Worst (largest |value|) step for a custom study, over all lids or just one ('Lid 3').

    Returns (step_name, value) or (None, None) if the study table is missing in every step.
    """
    column = component_column(component)
    best_step, best_value = None, -np.inf

    for step in step_names:
        path = os.path.join(base_folder, step, study_name, CUSTOM_TABLE_NAME)
        if not os.path.exists(path):
            continue
        df = read_lid_table(path)
        if lid is not None:
            df = df[df["Lid"] == lid]
        if df.empty:
            continue
        value = float(df[column].abs().max())
        if value > best_value:
            best_step, best_value = step, value

    if best_step is None:
        return None, None
    return best_step, best_value
