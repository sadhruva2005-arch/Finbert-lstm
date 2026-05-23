"""
Fuzzy logic feature engineering using skfuzzy.
Converts crisp scaled features into membership values for
'low', 'medium', and 'high' fuzzy sets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl

UNIVERSE = np.arange(-3.5, 3.5, 0.1)
MF_NAMES = ["low", "medium", "high"]


def build_fuzzy_variable(universe: np.ndarray = UNIVERSE) -> ctrl.Antecedent:
    """
    Create a fuzzy Antecedent with three triangular membership functions.

    Returns:
        skfuzzy Antecedent object with 'low', 'medium', 'high' MFs.
    """
    fuzzy_input = ctrl.Antecedent(universe, "input")
    fuzzy_input["low"] = fuzz.trimf(universe, [-3.5, -3.5, 0.0])
    fuzzy_input["medium"] = fuzz.trimf(universe, [-3.5, 0.0, 3.5])
    fuzzy_input["high"] = fuzz.trimf(universe, [0.0, 3.5, 3.5])
    return fuzzy_input


def fuzzify_series(
    values: np.ndarray,
    fuzzy_var: ctrl.Antecedent,
    universe: np.ndarray = UNIVERSE,
) -> pd.DataFrame:
    """
    Compute membership values for a 1-D array of crisp values.

    Returns:
        pd.DataFrame with columns ['low', 'medium', 'high'].
    """
    return pd.DataFrame(
        {
            mf: fuzz.interp_membership(universe, fuzzy_var[mf].mf, values)
            for mf in MF_NAMES
        }
    )


def fuzzify_dataframe(
    df: pd.DataFrame,
    feature_cols: list[str],
    universe: np.ndarray = UNIVERSE,
) -> pd.DataFrame:
    """
    Fuzzify multiple feature columns into membership-value columns.

    Args:
        df: DataFrame with crisp (scaled) feature values.
        feature_cols: Columns to fuzzify.
        universe: Fuzzy universe of discourse.

    Returns:
        pd.DataFrame with columns like 'volume_low', 'volume_medium', etc.
    """
    fuzzy_var = build_fuzzy_variable(universe)
    parts = []
    for col in feature_cols:
        mf_df = fuzzify_series(df[col].values, fuzzy_var, universe)
        mf_df.columns = [f"{col}_{mf}" for mf in MF_NAMES]
        mf_df.index = df.index
        parts.append(mf_df)
    return pd.concat(parts, axis=1)
