from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List, Set
import pandas as pd
import numpy as np


@dataclass
class TabularOmicsConfig:
    sample_id_col: str = "Sample ID"
    label_col: str = "Diagnosis"
    label_source: str = "first"          # "first" or "per_omics" (currently we validate all match)
    strict_sample_match: bool = True     # True: require identical sample sets
    drop_duplicates: str = "error"       # "error" | "first" | "last"
    coerce_numeric: bool = True          # convert feature values to float where possible
    nan_policy: str = "keep"             # "keep" | "zero" | "drop_feature" | "drop_sample"
    clinical_drop_cols: List[str] = None   # columns to drop before feature extraction per-omics


def _handle_duplicates(df: pd.DataFrame, sample_id_col: str, mode: str, verbose: bool = True) -> pd.DataFrame:
    dup = df[sample_id_col].duplicated(keep=False)
    if not dup.any():
        if verbose:
            print(f"  - No duplicated '{sample_id_col}' values found.")
        return df

    dups = df.loc[dup, sample_id_col].astype(str).unique().tolist()
    if verbose:
        print(f"  - Found {len(dups)} duplicated sample IDs in '{sample_id_col}'. Example: {dups[:5]}")

    if mode == "error":
        raise ValueError(f"Duplicate sample IDs found in column '{sample_id_col}': {dups[:10]} (showing up to 10)")
    if mode == "first":
        return df.drop_duplicates(subset=[sample_id_col], keep="first")
    if mode == "last":
        return df.drop_duplicates(subset=[sample_id_col], keep="last")
    raise ValueError(f"Invalid drop_duplicates='{mode}'. Use 'error', 'first', or 'last'.")


def _apply_nan_policy(
    df: pd.DataFrame,
    feature_cols: List[str],
    policy: str,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    if verbose:
        n_nan = int(df[feature_cols].isna().sum().sum()) if len(feature_cols) else 0
        print(f"  - NaN policy='{policy}'. Total NaNs across features: {n_nan}")

    if policy == "keep":
        return df, feature_cols

    if policy == "zero":
        df[feature_cols] = df[feature_cols].fillna(0.0)
        return df, feature_cols

    if policy == "drop_feature":
        good = [c for c in feature_cols if not df[c].isna().any()]
        if verbose:
            print(f"  - Dropping {len(feature_cols) - len(good)} features due to NaNs.")
        return df, good

    if policy == "drop_sample":
        before = df.shape[0]
        mask = ~df[feature_cols].isna().any(axis=1)
        df2 = df.loc[mask].copy()
        after = df2.shape[0]
        if verbose:
            print(f"  - Dropped {before - after} samples due to NaNs in features.")
        return df2, feature_cols

    raise ValueError("nan_policy must be one of: 'keep', 'zero', 'drop_feature', 'drop_sample'")


def read_omics_csv(
    file_path: str,
    omic_type: str,
    cfg: TabularOmicsConfig,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Reads one omics CSV, validates columns, handles duplicates,
    renames feature columns based on omic type, and returns dataframe.
    """

    if verbose:
        print(f"\n[READ] Omic='{omic_type}' | file='{file_path}'")

    df = pd.read_csv(file_path)

    if verbose:
        print(f"  - Loaded shape: {df.shape} (rows, cols)")
        print(f"  - First 5 columns: {list(df.columns[:5])}")

    # ---------------------------------------------------------
    # Validate required columns
    # ---------------------------------------------------------
    if cfg.sample_id_col not in df.columns:
        raise KeyError(f"[{omic_type}] Missing sample_id_col '{cfg.sample_id_col}' in {file_path}")
    if cfg.label_col not in df.columns:
        raise KeyError(f"[{omic_type}] Missing label_col '{cfg.label_col}' in {file_path}")

    # ---------------------------------------------------------
    # Handle duplicates
    # ---------------------------------------------------------
    df = _handle_duplicates(df, cfg.sample_id_col, cfg.drop_duplicates, verbose=verbose)

    # Ensure sample IDs are strings
    df[cfg.sample_id_col] = df[cfg.sample_id_col].astype(str)

    # ---------------------------------------------------------
    # Rename feature columns based on omic type
    # ---------------------------------------------------------

    # Columns to exclude from renaming
    columns_to_exclude = (
        [cfg.sample_id_col, cfg.label_col] +
        cfg.clinical_drop_cols
    )

    # Suffix mapping
    feature_suffix_map = {
        "miRNA": "",
        "gene": "",
        "transcript": "_transcript",
        "protein": "_protein",
    }

    suffix = feature_suffix_map.get(omic_type, "")

    # Identify feature columns
    feature_columns = [
        col for col in df.columns
        if col not in columns_to_exclude
    ]

    # Create rename mapping
    rename_map = {
        col: f"{col}{suffix}"
        for col in feature_columns
    }

    df = df.rename(columns=rename_map)

    if verbose:
        print(f"  - Applied suffix '{suffix}' to {len(rename_map)} feature columns")

    # ---------------------------------------------------------
    # Debug info
    # ---------------------------------------------------------
    if verbose:
        unique_n = df[cfg.sample_id_col].nunique()
        print(f"  - Unique sample IDs: {unique_n}")
        print(f"  - Label distribution:\n{df[cfg.label_col].value_counts(dropna=False).head(5)}")

    return df


def _coerce_features_to_numeric(df: pd.DataFrame, feature_cols: List[str], verbose: bool = True) -> pd.DataFrame:
    """Convert features to numeric where possible; non-convertible become NaN."""
    if not feature_cols:
        return df

    if verbose:
        print(f"  - Coercing {len(feature_cols)} feature columns to numeric (errors -> NaN).")

    before_non_null = int(df[feature_cols].notna().sum().sum())
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    after_non_null = int(df[feature_cols].notna().sum().sum())

    if verbose:
        print(f"  - Non-null feature entries before: {before_non_null}, after coercion: {after_non_null}")

    return df


def validate_sample_ids_across_omics(
    omics_dfs: Dict[str, pd.DataFrame],
    sample_id_col: str,
    strict: bool = True,
    verbose: bool = True,
) -> List[str]:
    """Checks sample ID consistency across all omics and returns the agreed sample_id list."""
    sets: Dict[str, Set[str]] = {
        omic: set(df[sample_id_col].astype(str).tolist())
        for omic, df in omics_dfs.items()
    }

    if verbose:
        print("\n[CHECK] Sample ID overlap across omics:")
        for omic, s in sets.items():
            print(f"  - {omic}: {len(s)} samples")

    omics = list(sets.keys())
    base = sets[omics[0]]

    if strict:
        for omic in omics[1:]:
            if sets[omic] != base:
                only_in_base = sorted(list(base - sets[omic]))[:10]
                only_in_other = sorted(list(sets[omic] - base))[:10]
                raise ValueError(
                    f"Sample IDs mismatch between '{omics[0]}' and '{omic}'.\n"
                    f"Only in '{omics[0]}' (up to 10): {only_in_base}\n"
                    f"Only in '{omic}' (up to 10): {only_in_other}"
                )
        agreed = sorted(list(base))
        if verbose:
            print(f"  ✅ Strict match: all omics share identical sample set ({len(agreed)} samples).")
        return agreed

    inter = set.intersection(*sets.values()) if sets else set()
    if len(inter) == 0:
        raise ValueError("No shared sample IDs across omics (intersection is empty).")
    agreed = sorted(list(inter))
    if verbose:
        print(f"  ✅ Non-strict match: using intersection ({len(agreed)} shared samples).")
    return agreed


def build_sample_store_from_configs(
    omics_config: Dict[str, Dict[str, str]],
    cfg: Optional[TabularOmicsConfig] = None,
    verbose: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    omics_config format:
      {"omics": {"gene": "...csv", "protein": "...csv", ...}}

    Returns:
      {sample_id: {"label": label_value, "data": {omic_type: {feature: value, ...}}}}
    """
    if cfg is None:
        cfg = TabularOmicsConfig()

    if verbose:
        print("\n==============================")
        print("[START] Building sample store")
        print("==============================")
        print(f"omics_config: sample_id_col='{cfg.sample_id_col}', label_col='{cfg.label_col}', "
              f"strict_sample_match={cfg.strict_sample_match}, nan_policy='{cfg.nan_policy}', "
              f"coerce_numeric={cfg.coerce_numeric}, drop_duplicates='{cfg.drop_duplicates}'")
        print(f"Omics provided: {list(omics_config.keys())}")

    # 1) Read all omics CSVs
    omics_dfs: Dict[str, pd.DataFrame] = {}
    for omic_type, path in omics_config.items():
        omics_dfs[omic_type] = read_omics_csv(path, omic_type, cfg, verbose=verbose)

    # 2) Validate sample IDs across omics
    sample_ids = validate_sample_ids_across_omics(
        omics_dfs,
        sample_id_col=cfg.sample_id_col,
        strict=cfg.strict_sample_match,
        verbose=verbose,
    )

    # 3) Label reference and validation
    omic_order = list(omics_dfs.keys())
    ref_omic = omic_order[0]
    label_ref_df = omics_dfs[ref_omic].set_index(cfg.sample_id_col)

    ref_labels = label_ref_df.loc[sample_ids, cfg.label_col].to_dict()

    if verbose:
        unique_labels = pd.Series(list(ref_labels.values())).value_counts(dropna=False)
        print(f"\n[LABEL] Using '{ref_omic}' as label reference.")
        print(f"  - Label distribution:\n{unique_labels}")

    for omic_type in omic_order[1:]:
        df = omics_dfs[omic_type].set_index(cfg.sample_id_col)
        other_labels = df.loc[sample_ids, cfg.label_col].to_dict()
        mismatches = 0
        for sid in sample_ids:
            a, b = ref_labels[sid], other_labels[sid]
            if pd.isna(a) and pd.isna(b):
                continue
            if a != b:
                mismatches += 1
                if mismatches <= 5 and verbose:
                    print(f"  - Label mismatch example | sample='{sid}' | {ref_omic}='{a}' vs {omic_type}='{b}'")
        if mismatches > 0:
            raise ValueError(f"Found {mismatches} label mismatches between '{ref_omic}' and '{omic_type}'.")

    # 4) Build store skeleton
    store: Dict[str, Dict[str, Any]] = {sid: {"label": ref_labels[sid], "data": {}} for sid in sample_ids}

    if verbose:
        print(f"\n[STORE] Initialized store for {len(sample_ids)} samples.")

    # 5) Fill store with per-omics feature dicts
    for omic_type, df in omics_dfs.items():
        if verbose:
            print(f"\n[OMICS] Processing omic='{omic_type}'")

        df = df.copy()
        df = df[df[cfg.sample_id_col].isin(sample_ids)].copy()
        df = df.set_index(cfg.sample_id_col).loc[sample_ids]

        # Drop clinical columns (so they never become "features")
        clinical_drop = set(cfg.clinical_drop_cols or [])
        clinical_drop = clinical_drop - {cfg.sample_id_col, cfg.label_col}  # never drop these
        
        if clinical_drop:
            drop_now = [c for c in clinical_drop if c in df.columns]
            if verbose and drop_now:
                print(f"  - Dropping clinical columns before features: {drop_now}")
            df = df.drop(columns=drop_now, errors="ignore")
        
        # Feature columns = everything except label (sample_id is already index)
        feature_cols = [c for c in df.columns if c != cfg.label_col]

        if verbose:
            print(f"  - Feature columns detected: {len(feature_cols)} (excluding label '{cfg.label_col}')")

        if cfg.coerce_numeric:
            df = _coerce_features_to_numeric(df, feature_cols, verbose=verbose)

        df, feature_cols = _apply_nan_policy(df, feature_cols, cfg.nan_policy, verbose=verbose)

        if verbose:
            print(f"  - Final feature columns used: {len(feature_cols)}")
            if len(feature_cols) > 0:
                # show a quick snapshot of ranges
                sample_vals = df[feature_cols].iloc[0].head(min(5, len(feature_cols)))
                print(f"  - Example features for first sample '{sample_ids[0]}' (first 5):\n{sample_vals}")

        # write feature dict per sample
        for sid in sample_ids:
            row = df.loc[sid, feature_cols]
            feats = row.to_dict() if isinstance(row, pd.Series) else {feature_cols[0]: float(row)}

            cleaned = {}
            for k, v in feats.items():
                if isinstance(v, (np.floating, float, int)) and not pd.isna(v):
                    cleaned[k] = float(v)
                else:
                    cleaned[k] = (None if pd.isna(v) else v)
            store[sid]["data"][omic_type] = cleaned

        if verbose:
            print(f"  ✅ Stored '{omic_type}' data for {len(sample_ids)} samples.")

    if verbose:
        # quick peek at one sample
        first_sid = sample_ids[0]
        print("\n==============================")
        print("[DONE] Store built successfully")
        print("==============================")
        print(f"Example sample: {first_sid}")
        print(f"  - label: {store[first_sid]['label']}")
        print(f"  - omics types: {list(store[first_sid]['data'].keys())}")
        for ot in store[first_sid]["data"].keys():
            print(f"    * {ot}: {len(store[first_sid]['data'][ot])} features")

    return store