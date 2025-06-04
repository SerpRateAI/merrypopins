"""
load_datasets.py
--------------

Read indentation experiment TXT data and metadata files into pandas DataFrames.
Provides:
  - load_txt: load a .txt data file (auto-detect columns) into a DataFrame, with header attrs
  - load_tdm: load a .tdm/.tdx metadata file (full channel list) into two DataFrames (root info and channel list)

Usage:
    from merrypopins.load_datasets import load_txt, load_tdm
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
          - Depth (nm): parsed from the first column with the name "Depth (nm)"
          - Load (µN): parsed from the second column with the name "Load (uN)"
          - Time (s): parsed from the third column with the name "Time (s)"
    Raises:
        FileNotFoundError: If the file does not exist.
        NotImplementedError: If the file type is not supported.
        UnicodeDecodeError: If the file cannot be decoded with UTF-8 or Latin-1.
        ValueError: If no numeric data is found in the file.
    """
    # Check if the file exists
    if not filepath.is_file():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    # Check file extension if it is a .txt file if not raise NotImplementedError
    if filepath.suffix.lower() != ".txt":
        raise NotImplementedError(
            f"File type '{filepath.suffix}' is not supported yet. Only '.txt' files are currently implemented."
        )

    # Read lines with encoding fallback
    try:
        raw = filepath.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        logger.warning(f"UTF-8 decode failed for {filepath}, falling back to Latin-1")
        try:
            raw = filepath.read_text(encoding="latin1")
        except Exception as e:
            logger.error(f"Latin-1 decode also failed for {filepath}.")
            raise e
    text = raw.splitlines()

    # Extract timestamp and num_points
    timestamp = None
    num_points = None
    for line in text:
        if timestamp is None and line.strip():
            timestamp = line.strip()
        if "Number of Points" in line and "=" in line:
            try:
                num_points = int(line.split("=", 1)[1])
            except ValueError:
                pass
        if timestamp and num_points is not None:
            break

    # Find start of numeric block: first row where every tab-split token is a number
    start_idx = None
    num_re = re.compile(r"^[-+]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$")
    for i, line in enumerate(text):
        tokens = line.strip().split("\t")
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
        col_names = text[header_idx].split("\t")
    else:
        col_names = []

    # Load the numeric block with tab delimiter
    data_str = "\n".join(text[start_idx:])
    arr = np.loadtxt(StringIO(data_str), delimiter="\t")

    # Force 2D array
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    # If header didn’t match, generate generic names
    if not col_names or len(col_names) != arr.shape[1]:
        col_names = [f"col_{i}" for i in range(arr.shape[1])]

    df = pd.DataFrame(arr, columns=col_names)
    df.attrs["timestamp"] = timestamp
    df.attrs["num_points"] = num_points
    logger.info(f"Loaded TXT data {filepath.name}: {df.shape[0]} × {df.shape[1]}")
    return df


def load_tdm(filepath: Path):
    """
    Load a .tdm metadata file into two DataFrames.
    Args:
        filepath: Path to the .tdm/.tdx file.
    Returns:
      - df_root: one row containing
          * name, description, title, author
          * every instance‐attribute under <instance_attributes>
      - df_channels: one row per <tdm_channel> with:
          group, channel_id, name, unit, description, datatype,
          sequence_id, block_id, block_length, value_type
    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not filepath.is_file():
        raise FileNotFoundError(f"TDM file not found: {filepath}")
    tree = ET.parse(str(filepath))
    root = tree.getroot()

    # --- extract tdm_root info ---
    tr = root.find(".//tdm_root")
    root_info = {
        "root_name": tr.findtext("name"),
        "root_description": tr.findtext("description"),
        "root_title": tr.findtext("title"),
        "root_author": tr.findtext("author"),
    }
    inst = tr.find("instance_attributes")
    for attr in inst:
        # double_attribute, string_attribute, long_attribute, time_attribute...
        key = attr.get("name")
        # string_attribute has <s> contents, others have .text
        if attr.tag.endswith("string_attribute"):
            val = attr.findtext("s")
        else:
            val = attr.text
        # strip leading/trailing whitespace (incl. newlines and tabs)
        val = val.strip() if isinstance(val, str) else val
        root_info[key] = val
    df_root = pd.DataFrame([root_info])

    # --- build helper maps for channels ---
    # 1) group id → group name
    group_map = {
        g.get("id"): g.findtext("name") for g in root.findall(".//tdm_channelgroup")
    }

    # 2) channel → sequence via localcolumn
    chan2seq = {}
    for lc in root.findall(".//localcolumn"):
        m1 = re.search(r'id\("([^"]+)"\)', (lc.findtext("measurement_quantity") or ""))
        m2 = re.search(r'id\("([^"]+)"\)', (lc.findtext("values") or ""))
        if m1 and m2:
            chan2seq[m1.group(1)] = m2.group(1)

    # --- now build per‐channel rows ---
    records = []
    for c in root.findall(".//tdm_channel"):
        cid = c.get("id")
        grp_txt = c.findtext("group") or ""
        m = re.search(r'id\("([^"]+)"\)', grp_txt)
        group = group_map.get(m.group(1)) if m else None

        seq = chan2seq.get(cid)
        # blk = seq2blk.get(seq)  # unused → removed
        rec = {
            "group": group,
            "channel_id": cid,
            "name": c.findtext("name"),
            "unit": c.findtext("unit_string"),
            "description": c.findtext("description"),
            "datatype": c.findtext("datatype"),
            "sequence_id": seq,
        }
        records.append(rec)

    df_channels = pd.DataFrame.from_records(records)
    logger.info(f"Loaded TDM metadata {filepath.name}: {len(df_channels)} channels")

    return df_root, df_channels


# package exports
__all__ = ["load_txt", "load_tdm"]
