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
import xml.etree.ElementTree as ET

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

    # Extract timestamp and num_points
    timestamp = None
    num_points = None
    for line in text:
        if timestamp is None and line.strip():
            timestamp = line.strip()
        if 'Number of Points' in line and '=' in line:
            try:
                num_points = int(line.split('=', 1)[1])
            except ValueError:
                pass
        if timestamp and num_points is not None:
            break

    # Find start of numeric block: first row where every tab-split token is a number
    start_idx = None
    num_re = re.compile(r'^[-+]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$')
    for i, line in enumerate(text):
        tokens = line.strip().split('\t')
        if tokens and all(num_re.match(tok) for tok in tokens):
            start_idx = i
            break
    if start_idx is None:
        raise ValueError(f"No numeric data found in {filepath}")

    # Header is the last non-empty line before the numeric block
    header_idx = start_idx - 1
    while header_idx >= 0 and not text[header_idx].strip():
        header_idx -= 1
    if header_idx >= 0:
        col_names = text[header_idx].split('\t')
    else:
        col_names = []

    # Load the numeric block with tab delimiter
    data_str = "\n".join(text[start_idx:])
    arr = np.loadtxt(StringIO(data_str), delimiter='\t')

    # Force 2D array
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    # If header didn’t match, generate generic names
    if not col_names or len(col_names) != arr.shape[1]:
        col_names = [f"col_{i}" for i in range(arr.shape[1])]

    df = pd.DataFrame(arr, columns=col_names)
    df.attrs['timestamp'] = timestamp
    df.attrs['num_points'] = num_points
    logger.info(f"Loaded TXT data {filepath.name}: {df.shape[0]} × {df.shape[1]}")
    return df


def load_tdm(filepath: Path) -> pd.DataFrame:
    """
    Load a .tdm metadata file into a DataFrame (name, unit, description, dtype).
    Only metadata is parsed.
    """
    if not filepath.is_file():
        raise FileNotFoundError(f"TDM file not found: {filepath}")

    # Parse the XML file
    tree = ET.parse(str(filepath))
    root = tree.getroot()

    # map channelgroup id -> name
    group_map = {}
    for g in root.findall('.//tdm_channelgroup'):
        gid = g.get('id')
        grp_name = g.findtext('name')
        group_map[gid] = grp_name

    records = []
    for c in root.findall('.//tdm_channel'):
        cid = c.get('id')
        name = c.findtext('name')
        unit = c.findtext('unit_string')
        dtype_val = c.findtext('datatype')
        desc = c.findtext('description')
        # extract group id
        grp_name = None
        grp_elem = c.find('group')
        if grp_elem is not None and grp_elem.text:
            m = re.search(r"id\(\"([^\"]+)\"\)", grp_elem.text)
            if m:
                grp_name = group_map.get(m.group(1))
        records.append({
            'group': grp_name,
            'channel_id': cid,
            'name': name,
            'unit': unit,
            'description': desc,
            'dtype': dtype_val,
            'data': None # Check why I added this line
        })

    df_meta = pd.DataFrame.from_records(records)
    logger.info(f"Loaded TDM metadata {filepath.name}: {len(df_meta)} channels")
    return df_meta

# package exports
__all__ = [
    'load_txt',
    'load_tdm'
]
