import pytest
from pathlib import Path
import pandas as pd

from indenter.load_datasets import txt_to_df, tdm_to_df, load_dataset


@pytest.fixture
def sample_txt(tmp_path):
    # Create a realistic .txt file with header, blank lines, column names, and 3 data rows
    content = (
        "Mon Apr 14 16:00:13 2025\n"
        "\n"
        "Number of Points = 3\n"
        "\n"
        "Depth (nm)\tLoad (uN)\tTime (s)\n"
        "-4.0\t2000.0\t0.0\n"
        "-5.0\t2100.0\t0.005\n"
        "-6.0\t2200.0\t0.010\n"
    )
    file = tmp_path / "sample.txt"
    file.write_text(content)
    return file


@pytest.fixture
def sample_idm(tmp_path):
    # Create a sample .idm/.tdm XML file with 3 metadata entries in a default namespace
    xml = (
        "<?xml version=\"1.0\"?>\n"
        "<root xmlns=\"http://example.com/schema\">\n"
        "  <localcolumns>\n"
        "    <localcolumn>\n"
        "      <name>Force</name>\n"
        "      <measurement_quantity>nN</measurement_quantity>\n"
        "      <sequence_representation>float</sequence_representation>\n"
        "    </localcolumn>\n"
        "    <localcolumn>\n"
        "      <name>Distance</name>\n"
        "      <measurement_quantity>nm</measurement_quantity>\n"
        "      <sequence_representation>float</sequence_representation>\n"
        "    </localcolumn>\n"
        "    <localcolumn>\n"
        "      <name>Time</name>\n"
        "      <measurement_quantity>s</measurement_quantity>\n"
        "      <sequence_representation>float</sequence_representation>\n"
        "    </localcolumn>\n"
        "  </localcolumns>\n"
        "</root>"
    )
    file = tmp_path / "sample.idm"
    file.write_text(xml)
    return file


def test_txt_to_df(sample_txt):
    df = txt_to_df(sample_txt)
    assert isinstance(df, pd.DataFrame)
    # Should load 3 rows and 3 columns of numeric data
    assert df.shape == (3, 3)
    # Timestamp and point count parsed correctly
    assert df.attrs['timestamp'] == "Mon Apr 14 16:00:13 2025"
    assert df.attrs['Number of Points'] == 3
    # Numeric values match input
    assert pytest.approx(df.iloc[0, 0]) == -4.0
    assert pytest.approx(df.iloc[1, 1]) == 2100.0
    assert pytest.approx(df.iloc[2, 2]) == 0.010


def test_tdm_to_df(sample_idm):
    df_meta = tdm_to_df(sample_idm)
    assert isinstance(df_meta, pd.DataFrame)
    # Should have exactly 3 metadata entries
    assert len(df_meta) == 3
    assert list(df_meta['name']) == ['Force', 'Distance', 'Time']
    assert list(df_meta['unit']) == ['nN', 'nm', 's']
    assert list(df_meta['dtype']) == ['float', 'float', 'float']


def test_load_dataset(sample_txt, sample_idm):
    df_data = txt_to_df(sample_txt)
    df_meta = tdm_to_df(sample_idm)
    df = load_dataset(df_data, df_meta)
    # Columns should be renamed according to metadata names
    assert list(df.columns) == ['Force', 'Distance', 'Time']
    # Units and dtypes stored in attrs
    assert df.attrs['units'] == ['nN', 'nm', 's']
    assert df.attrs['dtypes'] == ['float', 'float', 'float']
    # Data alignment preserved
    assert pytest.approx(df['Force'].iloc[2]) == -6.0
    assert pytest.approx(df['Distance'].iloc[2]) == 2200.0
    assert pytest.approx(df['Time'].iloc[2]) == 0.010
