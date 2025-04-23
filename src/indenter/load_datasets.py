"""
load_datasets.py
--------------

Read indentation experiment TXT data and metadata files into pandas DataFrames.
Provides:
  - load_txt: load a .txt data file (auto-detect columns) into a DataFrame, with header attrs
  - load_tdm: load a .tdm/.tdx metadata file (full channel list) into a DataFrame
  - merge_metadata: attach units and dtypes from metadata to the data DataFrame

Usage:
    from indenter.load_datasets import load_txt, load_tdm, merge_metadata

"""
import logging
from pathlib import Path
import re
from io import StringIO

import numpy as np
import pandas as pd
from nptdms import TdmsFile

# Module logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def load_txt(filepath: Path) -> pd.DataFrame:
    """
    Load a .txt indentation data file into a DataFrame.
    Automatically detects the header line (column names) and numeric block.

    Attempts UTF-8 decoding first, falls back to Latin-1 on failure.

    Args:
        filepath: Path to the .txt file.
    Returns:
        DataFrame with columns from the file, and attrs:
          - timestamp: first non-empty line
          - num_points: parsed from 'Number of Points = N'
    """
    if not filepath.is_file():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    # Read lines with encoding fallback
    try:
        raw = filepath.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        logger.warning(f"UTF-8 decode failed for {filepath}, falling back to Latin-1")
        raw = filepath.read_text(encoding='latin1')
    text = raw.splitlines()

    # extract timestamp and num_points
    timestamp = None
    num_points = None
    for line in text:
        if not timestamp and line.strip():
            timestamp = line.strip()
        if 'Number of Points' in line and '=' in line:
            try:
                num_points = int(line.split('=', 1)[1])
            except ValueError:
                continue
        if timestamp and num_points is not None:
            break

    # find start of numeric block (first all-numeric row)
    start_idx = None
    for i, line in enumerate(text):
        tokens = re.split(r"\s+|\t", line.strip())
        if tokens and all(re.match(r"^-?\d+(?:\.\d+)?$", tok) for tok in tokens):
            start_idx = i
            break
    if start_idx is None:
        raise ValueError(f"No numeric data found in {filepath}")

    # header is the last non-empty line before start_idx
    header_idx = start_idx - 1
    while header_idx >= 0 and not text[header_idx].strip():
        header_idx -= 1
    col_names = re.split(r"\s+|\t", text[header_idx].strip()) if header_idx >= 0 else []

    # load numeric data
    data_str = "\n".join(text[start_idx:])
    arr = np.loadtxt(StringIO(data_str))
    # ensure 2D array: scalar->(1,1), 1D->(n,1)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    df = pd.DataFrame(arr, columns=col_names)
    # store metadata
    df.attrs['timestamp'] = timestamp
    df.attrs['num_points'] = num_points
    logger.info(f"Loaded TXT data {filepath.name}: {df.shape[0]} rows, {df.shape[1]} cols")
    return df


def load_tdm(filepath: Path) -> pd.DataFrame:
    """
    Load a full TDM/TDX metadata file via nptdms into a DataFrame.
    Each row corresponds to one channel, including its data array and properties.

    Args:
        filepath: Path to the .tdm/.tdx file.
    Returns:
        DataFrame with one row per channel, columns:
          ['group','channel','unit','description','dtype','data', ...other props]
    """
    if not filepath.is_file():
        raise FileNotFoundError(f"TDM file not found: {filepath}")

    tdms = TdmsFile.read(str(filepath))
    records = []
    for group in tdms.groups():
        for chan_name in tdms[group].channels():
            ch = tdms[group][chan_name]
            props = dict(ch.properties)
            unit = props.pop('unit_string', None)
            desc = props.pop('description', None)
            rec = {
                'group': group,
                'channel': chan_name,
                'unit': unit,
                'description': desc,
                'dtype': str(ch.dtype),
                'data': ch[:],
            }
            rec.update(props)
            records.append(rec)

    df_meta = pd.DataFrame.from_records(records)
    logger.info(f"Loaded TDM metadata {filepath.name}: {len(df_meta)} channels")
    return df_meta


def merge_metadata(df: pd.DataFrame, df_meta: pd.DataFrame) -> pd.DataFrame:
    """
    Attach units and dtypes from metadata to a data DataFrame.
    Matches by column name (case-insensitive, alphanumeric only).

    Args:
        df: DataFrame with columns matching df_meta['channel'] (or ['name']).
        df_meta: metadata DataFrame from load_tdm.
    Returns:
        The same df, with attrs['units'] and attrs['dtypes'] as dicts.
    """
    # canonicalize names
    def canon(s):
        return re.sub(r"[^0-9a-z]", "", s.lower())

    key_col = 'channel' if 'channel' in df_meta.columns else 'name'
    meta_map = { canon(str(r[key_col])): r for _, r in df_meta.iterrows() }

    units = {}
    dtypes = {}
    for col in df.columns:
        c = canon(col)
        if c in meta_map:
            units[col] = meta_map[c].get('unit')
            dtypes[col] = meta_map[c].get('dtype')
        else:
            logger.warning(f"No metadata for column '{col}'")
    df.attrs['units'] = units
    df.attrs['dtypes'] = dtypes
    return df


# package exports
__all__ = [
    'load_txt',
    'load_tdm',
    'merge_metadata',
]
