#!/usr/bin/env python3
"""
load_datasets.py: Load indentation experiment data and metadata separately into pandas DataFrames.

Usage:
    from indenter import txt_to_df, tdm_to_df, load_dataset, load_tdm_full
"""
import logging
from pathlib import Path
import xml.etree.ElementTree as ET
from nptdms import TdmsFile
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from io import StringIO

# -----------------------------
# Module-level logger setup
# -----------------------------
enabled = True  # Toggle logging on/off
level = logging.INFO
if enabled:
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_txt(filepath: Path) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    Internal: Read header and numeric data from a .txt file.

    Args:
        filepath: Path to the .txt data file.
    Returns:
        header: Dictionary containing 'timestamp' and any key-value headers.
        data: Numeric numpy array of the loaded data.
    Raises:
        FileNotFoundError: if file does not exist.
        ValueError: if no numeric data can be found.
    """
    if not filepath.is_file():
        logger.error("Data file not found: %s", filepath)
        raise FileNotFoundError(f"Missing data file: {filepath}")

    header: Dict[str, Any] = {}
    # Read all lines
    lines = filepath.read_text().splitlines()

    # Timestamp: first non-empty line
    for line in lines:
        if line.strip():
            header['timestamp'] = line.strip()
            break

    # Number of points: first "key=val"
    for line in lines:
        if '=' in line:
            key, val = line.split('=', 1)
            header[key.strip()] = int(val.strip())
            break

    # Find numeric block start
    start_idx = None
    for i, line in enumerate(lines):
        if line.strip():
            tokens = line.strip().split()
            try:
                float(tokens[0])
                start_idx = i
                break
            except ValueError:
                continue
    if start_idx is None:
        logger.error("No numeric data found in %s", filepath)
        raise ValueError(f"No numeric data in {filepath}")

    # Load numeric data
    data_str = "\n".join(lines[start_idx:])
    data = np.loadtxt(StringIO(data_str))

    logger.debug("Loaded %d rows from %s", data.shape[0], filepath)
    return header, data


def txt_to_df(filepath: Path) -> pd.DataFrame:
    """
    Convert a .txt data file into a pandas DataFrame.

    Args:
        filepath: Path to the .txt data file.

    Returns:
        DataFrame: Numeric data, columns named col_0, col_1, ..., metadata in df.attrs.
    """
    header, data = load_txt(filepath)
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    col_names = [f"col_{i}" for i in range(data.shape[1])]
    df = pd.DataFrame(data, columns=col_names)
    df.attrs.update(header)
    logger.info("Text DataFrame created with shape %s", df.shape)
    return df


def load_tdm(filepath: Path) -> List[Dict[str, Any]]:
    """
    Internal: Parse .tdm/.idm XML file for basic column metadata: name, unit, dtype.

    Args:
        filepath: Path to the .tdm/.idm file.
    Returns:
        List of dicts, each with keys: 'name', 'unit', 'dtype'.
    Raises:
        FileNotFoundError: if file does not exist.
    """
    if not filepath.is_file():
        logger.error("Metadata file not found: %s", filepath)
        raise FileNotFoundError(f"Missing metadata file: {filepath}")

    tree = ET.parse(str(filepath))
    root = tree.getroot()
    ns_uri = root.tag.partition('}')[0].lstrip('{')
    ns = {'ns': ns_uri} if ns_uri else {}

    cols: List[Dict[str, Any]] = []
    for elem in root.findall('.//ns:localcolumn', ns):
        name = elem.findtext('ns:name', default=None, namespaces=ns)
        unit = elem.findtext('ns:measurement_quantity', default=None, namespaces=ns)
        dtype = elem.findtext('ns:sequence_representation', default=None, namespaces=ns)
        cols.append({
            'name': name.strip() if name else None,
            'unit': unit,
            'dtype': dtype
        })
    logger.debug("Parsed %d metadata entries from %s", len(cols), filepath)
    return cols


def tdm_to_df(filepath: Path) -> pd.DataFrame:
    """
    Convert a .tdm/.idm metadata file into a pandas DataFrame.

    Args:
        filepath: Path to metadata file.
    Returns:
        DataFrame with columns ['name', 'unit', 'dtype'].
    """
    cols = load_tdm(filepath)
    df = pd.DataFrame(cols)
    logger.info("Metadata DataFrame created with %d entries", len(df))
    return df


def load_dataset(df_data: pd.DataFrame, df_meta: pd.DataFrame) -> pd.DataFrame:
    """
    Merge data and metadata DataFrames into one DataFrame.

    Args:
        df_data: DataFrame of raw numeric data.
        df_meta: DataFrame of metadata (name, unit, dtype).
    Returns:
        DataFrame: renamed according to metadata, only columns present in metadata.
    """
    # Map generic columns to metadata names
    names = df_meta['name'].fillna('').tolist()
    col_map = {old: new for old, new in zip(df_data.columns, names) if new}

    # Keep only columns that have metadata
    df = df_data.loc[:, list(col_map.keys())].rename(columns=col_map)
    df.attrs['units'] = df_meta['unit'].tolist()
    df.attrs['dtypes'] = df_meta['dtype'].tolist()
    logger.info("Merged DataFrame shape %s", df.shape)
    return df


def load_tdm_full(filepath: Path) -> pd.DataFrame:
    """
    Load full TDM/TDX metadata and data into a pandas DataFrame (one row per channel).

    Args:
        filepath: Path to .tdm or .tdx file.
    Returns:
        DataFrame where each row corresponds to a channel with its data and properties.
    Raises:
        FileNotFoundError: if file does not exist.
        ImportError: if nptdms is unavailable.
    """
    # Dependency guard
    if TdmsFile is None:
        raise ImportError("nptdms is required for load_tdm_full. Install with `pip install nptdms`.")

    if not filepath.is_file():
        logger.error("TDM file not found: %s", filepath)
        raise FileNotFoundError(f"Missing TDM file: {filepath}")

    tdms = TdmsFile.read(str(filepath))
    records: List[Dict[str, Any]] = []
    for group in tdms.groups():
        for channel_name in tdms[group].channels():
            ch = tdms[group][channel_name]
            props = ch.properties.copy()
            unit = props.pop('unit_string', None)
            desc = props.pop('description', None)
            rec = {
                'group': group,
                'channel': channel_name,
                'unit': unit,
                'description': desc,
                'dtype': str(ch.dtype),
                'data': ch[:],
            }
            rec.update(props)
            records.append(rec)

    df_full = pd.DataFrame.from_records(records)
    logger.info("Loaded full TDM with %d channels", len(df_full))
    return df_full

# -----------------------------
# CLI entrypoint
# -----------------------------
if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Load text and metadata files into DataFrames (basic or full)."
    )
    parser.add_argument('data_file', type=Path, help='Path to the .txt data file')
    parser.add_argument('meta_file', type=Path, help='Path to the .tdm/.idm metadata file')
    parser.add_argument('--full', action='store_true', help='Load full TDM/TDX with nptdms')
    args = parser.parse_args()

    # Attempt to load text data, skip on error
    try:
        df_data = txt_to_df(args.data_file)
        print("Text DataFrame:")
        print(df_data.head())
    except Exception as e:
        print(f"Skipping data load: {e}")
        df_data = None

    # Load metadata per mode
    if args.full:
        try:
            df_meta = load_tdm_full(args.meta_file)
            print(f"Loaded full TDM with {len(df_meta)} channels")
        except Exception as e:
            print(f"Failed full TDM load: {e}")
            sys.exit(1)
    else:
        try:
            df_meta = tdm_to_df(args.meta_file)
            print("Metadata DataFrame:")
            print(df_meta.head())
        except Exception as e:
            print(f"Failed metadata load: {e}")
            sys.exit(1)
