"""
load_datasets.py: Load indentation experiment data and metadata separately into pandas DataFrames.

Usage:
    from indenter import txt_to_df, tdm_to_df, load_dataset
"""
import logging
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from io import StringIO

# Configure module-level logger
enabled = True
level = logging.INFO
if enabled:
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_txt(filepath: Path) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    Internal: Read header and numeric data from a .txt file.
    """
    if not filepath.is_file():
        logger.error("Data file not found: %s", filepath)
        raise FileNotFoundError(f"Missing data file: {filepath}")

    header: Dict[str, Any] = {}
    # Read all lines
    lines = filepath.read_text().splitlines()

    # Parse timestamp (first non-empty line)
    for line in lines:
        if line.strip():
            header['timestamp'] = line.strip()
            break

    # Parse number of points (first line containing '=')
    for line in lines:
        if '=' in line:
            key, val = line.split('=', 1)
            header[key.strip()] = int(val.strip())
            break

    # Find index of first numeric data line
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
    n_cols = data.shape[1]
    col_names = [f"col_{i}" for i in range(n_cols)]
    df = pd.DataFrame(data, columns=col_names)
    df.attrs.update(header)
    logger.info("Text DataFrame created with shape %s", df.shape)
    return df


def load_tdm(filepath: Path) -> List[Dict[str, Any]]:
    """
    Internal: Parse .tdm or .idm XML file for column metadata.
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
            'dtype': dtype,
        })
    logger.debug("Parsed %d metadata entries from %s", len(cols), filepath)
    return cols


def tdm_to_df(filepath: Path) -> pd.DataFrame:
    """
    Convert a .tdm or .idm metadata file into a pandas DataFrame.

    Args:
        filepath: Path to the .tdm/.idm metadata file.

    Returns:
        DataFrame: Columns ['name', 'unit', 'dtype'] describing each data series.
    """
    cols = load_tdm(filepath)
    df = pd.DataFrame(cols)
    logger.info("Metadata DataFrame created with %d entries", len(df))
    return df


def load_dataset(df_data: pd.DataFrame, df_meta: pd.DataFrame) -> pd.DataFrame:
    """
    Merge data and metadata DataFrames into a single labeled DataFrame.

    Args:
        df_data: DataFrame of numeric data (from txt_to_df).
        df_meta: DataFrame of metadata (from tdm_to_df).

    Returns:
        DataFrame: Numeric data with column names from metadata and metadata in attrs.
    """
    names = df_meta['name'].fillna('').tolist()
    col_map = {old: new for old, new in zip(df_data.columns, names) if new}
    df = df_data.rename(columns=col_map)
    df.attrs['units'] = df_meta['unit'].tolist()
    df.attrs['dtypes'] = df_meta['dtype'].tolist()
    logger.info("Merged DataFrame shape %s", df.shape)
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Load text and metadata files into separate DataFrames or merge them."
    )
    parser.add_argument('data_file', type=Path, help='Path to the .txt data file')
    parser.add_argument('meta_file', type=Path, help='Path to the .tdm/.idm metadata file')
    args = parser.parse_args()

    df_data = txt_to_df(args.data_file)
    df_meta = tdm_to_df(args.meta_file)
    print(df_data.head(), '\n')
    print(df_meta.head())
