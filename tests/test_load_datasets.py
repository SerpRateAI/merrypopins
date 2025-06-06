import pytest
from unittest.mock import patch
import logging
from pathlib import Path
import numpy as np
import pandas as pd

from merrypopins.load_datasets import load_txt, load_tdm

# ==========================
# Fixtures for test inputs
# ==========================


@pytest.fixture
def sample_txt(tmp_path):
    # A simple well‐formed TXT file: timestamp, "Number of Points", header row, and 3 lines of numeric data.
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
    p = tmp_path / "sample.txt"
    p.write_text(content)
    return p


@pytest.fixture
def sample_txt_latin(tmp_path):
    # A TXT file encoded in Latin-1 (with a special character) to force the UTF-8 fallback path.
    content = "Tést\n\nNumber of Points = 1\n\nVal\n1.0\n"
    p = tmp_path / "latin.txt"
    p.write_text(content, encoding="latin1")
    return p


@pytest.fixture
def sample_txt_invalid_num(tmp_path):
    # A TXT file where the first "Number of Points" is non‐numeric, then a valid one follows.
    content = (
        "TS\n\n"
        "Number of Points = abc\n"
        "Number of Points = 2\n"
        "\n"
        "X\n"
        "1.0\n"
        "2.0\n"
    )
    p = tmp_path / "invalid_num.txt"
    p.write_text(content)
    return p


@pytest.fixture
def sample_onecol_many(tmp_path):
    # A 1-column file to trigger the arr.ndim == 1 reshaping branch.
    content = "TS\n\n" "Number of Points = 3\n" "\n" "Val\n" "10.0\n" "20.0\n" "30.0\n"
    p = tmp_path / "onecolmany.txt"
    p.write_text(content)
    return p


# ==========================
# Tests for load_txt
# ==========================


def test_load_txt_file_not_found():
    # Should raise FileNotFoundError if the path does not exist
    with pytest.raises(FileNotFoundError):
        load_txt(Path("no_such.txt"))


def test_load_txt_not_implemented_for_other_types(tmp_path):
    # Should raise NotImplementedError if the file type is not .txt
    p = tmp_path / "file.csv"
    p.write_text("some,data\n1,2")
    with pytest.raises(
        NotImplementedError, match=r"File type '.csv' is not supported yet"
    ):
        load_txt(p)


def test_load_txt_encoding_fallback_failure(tmp_path, caplog):
    # Create a dummy file with invalid UTF-8 encoding and patch the read_text method to simulate encoding errors.
    # This will raise a UnicodeDecodeError when trying to read the file.
    # The second read_text call will raise a generic Exception to simulate a failure in Latin-1 decoding.
    # The test will check that the correct error messages are logged.
    dummy_path = tmp_path / "bad_encoding.txt"
    dummy_path.write_bytes(b"\xff")  # Invalid UTF-8 byte sequence

    with patch.object(
        Path,
        "read_text",
        side_effect=[
            UnicodeDecodeError("utf-8", b"", 0, 1, "invalid start byte"),
            Exception("Latin-1 also failed"),
        ],
    ):
        caplog.set_level(logging.DEBUG)
        with pytest.raises(Exception, match="Latin-1 also failed"):
            load_txt(dummy_path)

        messages = [r.message for r in caplog.records]
        assert any("UTF-8 decode failed" in m for m in messages)
        assert any("Latin-1 decode also failed" in m for m in messages)


def test_load_txt_no_numeric(tmp_path):
    # Should raise ValueError when no numeric rows are found
    p = tmp_path / "bad.txt"
    p.write_text("no numbers at all\njust text\n")
    with pytest.raises(ValueError):
        load_txt(p)


def test_load_txt_basic(sample_txt):
    # The basic successful case: DataFrame shape, timestamp, num_points, and a sample value check
    df = load_txt(sample_txt)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (3, 3)
    assert df.attrs["timestamp"] == "Mon Apr 14 16:00:13 2025"
    assert df.attrs["num_points"] == 3
    # second row, second column ("Load")
    assert np.isclose(df.iloc[1, 1], 2100.0)


def test_load_txt_encoding_fallback(sample_txt_latin, caplog):
    # Should log a warning and succeed via Latin-1 fallback
    caplog.set_level(logging.WARNING)
    df = load_txt(sample_txt_latin)
    assert df.shape == (1, 1)
    assert df.attrs["num_points"] == 1
    assert "falling back to Latin-1" in caplog.text


def test_load_txt_invalid_num(sample_txt_invalid_num):
    # First non‐numeric 'abc' should be skipped, then parse 2 correctly
    df = load_txt(sample_txt_invalid_num)
    assert df.attrs["num_points"] == 2
    assert list(df.columns) == ["X"]


def test_load_txt_ndim1(sample_onecol_many):
    # One-column file should be reshaped to shape (N,1) with correct header
    df = load_txt(sample_onecol_many)
    assert df.shape == (3, 1)
    assert list(df.columns) == ["Val"]
    assert np.isclose(df.iloc[2, 0], 30.0)


def test_load_txt_numeric_first_row(tmp_path):
    # If the first non‐blank line is numeric, header_idx < 0 → fallback to col_ names
    content = "1\t2\t3\n4\t5\t6\n"
    p = tmp_path / "numeric_first.txt"
    p.write_text(content)
    df = load_txt(p)
    assert list(df.columns) == ["col_0", "col_1", "col_2"]
    assert df.shape == (2, 3)
    assert np.allclose(df.values, np.array([[1, 2, 3], [4, 5, 6]]))


def test_load_txt_header_skip_blank(tmp_path):
    # This file has the header line, then two blank lines, then the numeric data.
    # We want to exercise the loop that skips blank lines to find the real header.
    content = (
        "Some Timestamp\n"
        "\n"
        "Number of Points = 2\n"
        "\n"
        "ColA\tColB\n"
        "\n"
        "\n"
        "1\t2\n"
        "3\t4\n"
    )
    p = tmp_path / "skip_blank_header.txt"
    p.write_text(content)

    df = load_txt(p)
    # After skipping the two blank lines, the header should be ['ColA', 'ColB']
    assert list(df.columns) == ["ColA", "ColB"]
    # Numeric block shape and values should still be correct
    assert df.shape == (2, 2)
    assert np.allclose(df.values, np.array([[1, 2], [3, 4]]))
    # And num_points should have been parsed correctly
    assert df.attrs["num_points"] == 2


# ==========================
# Tests for load_tdm
# ==========================


def test_load_tdm_file_not_found():
    # Should raise FileNotFoundError if the .tdm path does not exist
    with pytest.raises(FileNotFoundError):
        load_tdm(Path("missing.tdm"))


@pytest.mark.slow
def test_load_tdm_basic(tmp_path):
    # A minimal TDM XML: root metadata + one channel + one group, no sequence mapping
    xml = """<?xml version="1.0"?>
    <usi:tdm xmlns:usi="http://www.ni.com/Schemas/USI/1_0">
      <tdm_root id="R1">
        <name>RootName</name>
        <description>RootDesc</description>
        <title>RootTitle</title>
        <author>Auth</author>
        <instance_attributes>
          <double_attribute name="DA">3.14</double_attribute>
          <string_attribute name="SA"><s>foo</s></string_attribute>
          <long_attribute name="LA">0</long_attribute>
        </instance_attributes>
      </tdm_root>
      <tdm_channelgroup id="G1"><name>Grp1</name></tdm_channelgroup>
      <tdm_channel id="C1">
        <name>Ch1</name>
        <unit_string>u</unit_string>
        <datatype>DT_DBL</datatype>
        <description>desc1</description>
        <group>#xpointer(id("G1"))</group>
      </tdm_channel>
    </usi:tdm>
    """
    f = tmp_path / "basic.tdm"
    f.write_text(xml)
    df_root, df_ch = load_tdm(f)

    # Check root metadata DataFrame
    assert df_root.shape == (
        1,
        7,
    )  # root_name, root_description, root_title, root_author + 3 attrs
    r = df_root.iloc[0]
    assert r["root_name"] == "RootName"
    assert r["root_description"] == "RootDesc"
    assert r["root_title"] == "RootTitle"
    assert r["root_author"] == "Auth"
    assert float(r["DA"]) == pytest.approx(3.14)
    assert r["SA"] == "foo"
    assert r["LA"] == "0"

    # Check channel DataFrame
    assert df_ch.shape == (
        1,
        7,
    )  # group, channel_id, name, unit, description, datatype, sequence_id
    c = df_ch.iloc[0]
    assert c["channel_id"] == "C1"
    assert c["name"] == "Ch1"
    assert c["unit"] == "u"
    assert c["datatype"] == "DT_DBL"
    assert c["description"] == "desc1"
    assert c["group"] == "Grp1"
    # No <localcolumn> mapping present → sequence_id remains NaN
    assert pd.isna(c["sequence_id"])


@pytest.mark.slow
def test_load_tdm_sequence_mapping(tmp_path):
    # Now include a <usi:include> block, a <double_sequence>, and a <localcolumn> mapping C1 → s1
    xml = """<?xml version="1.0"?>
    <usi:tdm xmlns:usi="http://www.ni.com/Schemas/USI/1_0">
      <tdm_root id="R1">
        <name>X</name><description>Y</description>
        <title>Z</title><author>A</author>
        <instance_attributes/>
      </tdm_root>
      <usi:include>
        <file url="x.tdx">
          <block id="blk0" length="10" valueType="eFloat64Usi"/>
        </file>
      </usi:include>
      <usi:data>
        <double_sequence id="s1"><values external="blk0"/></double_sequence>
      </usi:data>
      <tdm_channelgroup id="G1"><name>G1</name></tdm_channelgroup>
      <tdm_channel id="C1">
        <name>C1</name><unit_string>u</unit_string>
        <datatype>DT_D</datatype><description>d</description>
        <group>#xpointer(id("G1"))</group>
      </tdm_channel>
      <localcolumn id="lc1">
        <measurement_quantity>#xpointer(id("C1"))</measurement_quantity>
        <values>#xpointer(id("s1"))</values>
      </localcolumn>
    </usi:tdm>
    """
    f = tmp_path / "map.tdm"
    f.write_text(xml)
    df_root, df_ch = load_tdm(f)

    # The localcolumn mapping should give sequence_id='s1' for channel C1
    assert df_ch.loc[0, "sequence_id"] == "s1"
