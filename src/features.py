import pandas as pd

BASE_FEATURES = ["u", "g", "r", "i", "z", "redshift"]
COLOR_FEATURES = ["u_g", "g_r", "r_i", "i_z"]


def add_color_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Add adjacent-band color index features.

    Parameters
    ----------
    df:
        DataFrame containing at least u, g, r, i, z.

    Returns
    -------
    pd.DataFrame
        A copied DataFrame with four extra color features.
    """
    required = ["u", "g", "r", "i", "z"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required photometric columns: {missing}")

    out = df.copy()
    out["u_g"] = out["u"] - out["g"]
    out["g_r"] = out["g"] - out["r"]
    out["r_i"] = out["r"] - out["i"]
    out["i_z"] = out["i"] - out["z"]
    return out


def get_feature_columns(use_color_features: bool = True) -> list[str]:
    """Return feature column names used by the model."""
    if use_color_features:
        return BASE_FEATURES + COLOR_FEATURES
    return BASE_FEATURES.copy()
