import pytest
import logging
from pathlib import Path
import numpy as np
import pandas as pd

import indenter.load_datasets as ld
from indenter.load_datasets import load_txt, load_tdm

# ==========================
# Fixtures for test inputs
# ==========================

@pytest.fixture
def sample_txt(tmp_path):
    # Simple TXT with clear headers and numeric block
    content = (
        "Mon Apr 14 16:00:13 2025\n"
        "\n"
        "Number of Points = 3\n"
        "\n"
        "Depth(nm)\tLoad(uN)\tTime(s)\n"
        "-4.0\t2000.0\t0.0\n"
        "-5.0\t2100.0\t0.005\n"
        "-6.0\t2200.0\t0.010\n"
    )
    file = tmp_path / "sample.txt"
    file.write_text(content)
    return file

@pytest.fixture
def sample_txt_latin(tmp_path):
    # TXT encoded in Latin-1 with special char to force fallback
    content = "Tést\n\nNumber of Points = 1\n\nVal\n1.0\n"
    file = tmp_path / "latin.txt"
    file.write_text(content, encoding='latin1')
    return file

@pytest.fixture
def sample_txt_invalid_num(tmp_path):
    # First invalid num_points, then valid
    content = (
        "TS\n\n"
        "Number of Points = abc\n"
        "Number of Points = 2\n"
        "\n"
        "X\n"
        "1.0\n"
        "2.0\n"
    )
    file = tmp_path / "invalid_num.txt"
    file.write_text(content)
    return file

@pytest.fixture
def sample_onecol_many(tmp_path):
    # One-column with multiple rows to hit arr.ndim == 1
    content = (
        "TS\n\n"
        "Number of Points = 3\n"
        "\n"
        "Val\n"
        "10.0\n"
        "20.0\n"
        "30.0\n"
    )
    file = tmp_path / "onecolmany.txt"
    file.write_text(content)
    return file

# ==========================
# Tests for load_txt
# ==========================

def test_load_txt_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_txt(Path('no_such.txt'))


def test_load_txt_no_numeric(tmp_path):
    bad = tmp_path / 'bad.txt'
    bad.write_text("No numbers here\n")
    with pytest.raises(ValueError):
        load_txt(bad)


def test_load_txt_basic(sample_txt):
    df = load_txt(sample_txt)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (3, 3)
    assert df.attrs['timestamp'] == "Mon Apr 14 16:00:13 2025"
    assert df.attrs['num_points'] == 3
    # the second row, second column "Load"
    assert np.isclose(df.iloc[1, 1], 2100.0)


def test_load_txt_encoding_fallback(sample_txt_latin, caplog):
    caplog.set_level(logging.WARNING)
    df = load_txt(sample_txt_latin)
    assert df.shape == (1, 1)
    assert df.attrs['num_points'] == 1
    assert "falling back to Latin-1" in caplog.text


def test_load_txt_invalid_num(sample_txt_invalid_num):
    df = load_txt(sample_txt_invalid_num)
    # Should skip the bad "abc" then parse 2
    assert df.attrs['num_points'] == 2
    assert list(df.columns) == ['X']


def test_load_txt_ndim1(sample_onecol_many):
    df = load_txt(sample_onecol_many)
    assert df.shape == (3, 1)
    assert list(df.columns) == ['Val']
    assert np.isclose(df.iloc[2, 0], 30.0)


def test_load_txt_missing_header(tmp_path):
    # Numeric block without any header line triggers error
    content = "1.0 2.0\n3.0 4.0\n"
    file = tmp_path / "noheader.txt"
    file.write_text(content)
    with pytest.raises(ValueError):
        load_txt(file)

# ==========================
# Tests for load_tdm
# ==========================

def test_load_tdm_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_tdm(Path('missing.tdm'))


def test_load_tdm_success(tmp_path):
    # Create a minimal .tdm XML with one channel in one group
    xml = """<?xml version="1.0"?>
            <usi:tdm xmlns:usi="http://www.ni.com/Schemas/USI/1_0">
            <tdm_channelgroup id="G1">
                <name>Test Group</name>
            </tdm_channelgroup>
            <tdm_channel id="C1">
                <name>Channel1</name>
                <unit_string>u</unit_string>
                <datatype>DT_DOUBLE</datatype>
                <description>desc</description>
                <group>#xpointer(id("G1"))</group>
            </tdm_channel>
            </usi:tdm>
        """
    f = tmp_path / "file.tdm"
    f.write_text(xml)
    df_meta = load_tdm(f)

    # Should load exactly one channel
    assert df_meta.shape[0] == 1

    row = df_meta.iloc[0]
    # New columns are 'channel_id' not 'channel'
    assert row['channel_id'] == 'C1'
    assert row['name'] == 'Channel1'
    assert row['unit'] == 'u'
    assert row['description'] == 'desc'
    assert row['dtype'] == 'DT_DOUBLE'
    assert row['group'] == 'Test Group'
