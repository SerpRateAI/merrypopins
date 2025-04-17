import pytest
import sys
import runpy
from pathlib import Path
import pandas as pd
import numpy as np

import indenter.load_datasets as mod
from indenter.load_datasets import (
    txt_to_df,
    tdm_to_df,
    load_dataset,
    load_tdm_full,
    load_txt
)

# ==========================
# Fixtures for test inputs
# ==========================

@pytest.fixture
# Creates a sample .txt file with three numeric columns for testing basic loader
def sample_txt(tmp_path):
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
# Creates a .txt file lacking any numeric data to test error handling
def sample_txt_no_numeric(tmp_path):
    content = "Header only\nNo values here\n"
    file = tmp_path / "bad.txt"
    file.write_text(content)
    return file

@pytest.fixture
# Creates a one-column .txt file to verify reshape logic
def sample_onecol_txt(tmp_path):
    content = (
        "TimeStamp\n"
        "\n"
        "Number of Points = 2\n"
        "\n"
        "Val\n"
        "1.0\n"
        "2.0\n"
    )
    file = tmp_path / "onecol.txt"
    file.write_text(content)
    return file

@pytest.fixture
# Creates minimal TDM file with no <localcolumn> entries to test empty metadata
def sample_tdm_empty(tmp_path):
    xml = (
        "<?xml version='1.0'?><root>"
        "<localcolumns></localcolumns>"
        "</root>"
    )
    file = tmp_path / "empty.tdm"
    file.write_text(xml)
    return file

@pytest.fixture
# Creates simple TDM/IDM file containing two metadata columns
def sample_idm(tmp_path):
    xml = (
        "<?xml version='1.0'?><root xmlns='http://example.com/schema'>"
        "<localcolumns>"
        "<localcolumn><name>Force</name><measurement_quantity>nN</measurement_quantity><sequence_representation>float</sequence_representation></localcolumn>"
        "<localcolumn><name>Distance</name><measurement_quantity>nm</measurement_quantity><sequence_representation>float</sequence_representation></localcolumn>"
        "</localcolumns></root>"
    )
    file = tmp_path / "sample.idm"
    file.write_text(xml)
    return file

# ==========================================
# Tests for load_txt and txt_to_df functions
# ==========================================

def test_load_txt_file_not_found():
    """
    Loading a non-existent .txt should raise FileNotFoundError
    """
    with pytest.raises(FileNotFoundError):
        load_txt(Path('does_not_exist.txt'))


def test_load_txt_no_numeric(sample_txt_no_numeric):
    """
    Loading a .txt without numeric data should raise ValueError
    """
    with pytest.raises(ValueError):
        load_txt(sample_txt_no_numeric)


def test_onecol_txt_reshape(sample_onecol_txt):
    """
    When .txt has only one column, txt_to_df should reshape to single-column DataFrame
    """
    df = txt_to_df(sample_onecol_txt)
    assert df.shape == (2, 1)
    assert list(df.columns) == ['col_0']
    assert pytest.approx(df.iloc[1, 0]) == 2.0


def test_txt_to_df(sample_txt):
    """
    Verify txt_to_df produces correct DataFrame shape, header attrs, and values
    """
    df = txt_to_df(sample_txt)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (3, 3)
    assert df.attrs['timestamp'] == "Mon Apr 14 16:00:13 2025"
    assert df.attrs['Number of Points'] == 3
    assert pytest.approx(df.iloc[0, 0]) == -4.0
    assert pytest.approx(df.iloc[1, 1]) == 2100.0
    assert pytest.approx(df.iloc[2, 2]) == 0.010

# =====================================
# Tests for tdm_to_df and metadata parsing
# =====================================

def test_tdm_to_df_file_not_found():
    """
    Parsing a missing TDM file should raise FileNotFoundError
    """
    with pytest.raises(FileNotFoundError):
        tdm_to_df(Path('missing.tdm'))


def test_tdm_to_df_empty(sample_tdm_empty):
    """
    An empty TDM should result in an empty DataFrame
    """
    df = tdm_to_df(sample_tdm_empty)
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_tdm_to_df(sample_idm):
    """
    Verify tdm_to_df extracts correct names, units, and dtypes
    """
    df_meta = tdm_to_df(sample_idm)
    assert list(df_meta['name']) == ['Force', 'Distance']
    assert list(df_meta['unit']) == ['nN', 'nm']
    assert list(df_meta['dtype']) == ['float', 'float']

# ==========================================
# Tests for merging data with metadata
# ==========================================

def test_load_dataset(sample_txt, sample_idm):
    """
    load_dataset should rename columns and attach metadata attributes
    """
    df_data = txt_to_df(sample_txt)
    df_meta = tdm_to_df(sample_idm)
    df = load_dataset(df_data, df_meta)
    assert list(df.columns) == ['Force', 'Distance']
    assert df.attrs['units'] == ['nN', 'nm']
    assert df.attrs['dtypes'] == ['float', 'float']
    assert pytest.approx(df['Force'].iloc[2]) == -6.0

# ====================================
# Tests for load_tdm_full function
# ====================================

def test_load_tdm_full_dependency(monkeypatch):
    """
    If TdmsFile is unset, load_tdm_full should raise ImportError
    """
    monkeypatch.setattr(mod, 'TdmsFile', None)
    with pytest.raises(ImportError):
        load_tdm_full(Path('dummy.tdm'))


def test_load_tdm_full_file_not_found():
    """
    If file missing, load_tdm_full should raise FileNotFoundError
    """
    with pytest.raises(FileNotFoundError):
        load_tdm_full(Path('not_here.tdm'))

# Dummy classes to simulate nptdms behavior
class DummyCh:
    """Simulates a TDMS channel with properties and data slice."""
    def __init__(self, name):
        self.dtype = 'DT_DOUBLE'
        self.properties = {'unit_string': 'u', 'description': 'desc', 'extra': 'val'}
    def __getitem__(self, idx):
        return np.array([1, 2, 3])

class DummyTDMS:
    """Simulates a TDMSFile object with one group and channel."""
    def __init__(self): pass
    def groups(self): return ['G1']
    def channels(self): return ['C1']
    def __getitem__(self, key):
        return DummyCh(key) if key != 'G1' else self

@pytest.fixture
# Provides a dummy .tdm file and patches TdmsFile.read to return DummyTDMS
def sample_tdm_full(monkeypatch, tmp_path):
    f = tmp_path / 'full.tdm'
    f.write_text('')
    monkeypatch.setattr(mod.TdmsFile, 'read', lambda path: DummyTDMS())
    return f


def test_load_tdm_full_success(sample_tdm_full):
    """
    load_tdm_full should return DataFrame with channel records
    """
    df_full = load_tdm_full(sample_tdm_full)
    assert isinstance(df_full, pd.DataFrame)
    assert df_full.shape[0] == 1
    row = df_full.iloc[0]
    assert row['group'] == 'G1'
    assert row['channel'] == 'C1'
    assert row['unit'] == 'u'
    assert row['description'] == 'desc'
    assert row['dtype'] == 'DT_DOUBLE'
    assert isinstance(row['data'], np.ndarray)
    assert row['extra'] == 'val'

# ====================================
# CLI entrypoint tests
# ====================================

def test_cli_default(tmp_path, monkeypatch, capsys):
    """
    Running module with default mode should print basic DataFrames
    """
    data = tmp_path / 'd.txt'
    data.write_text("T\n\nNumber of Points = 1\n\n5.0\n")
    meta = tmp_path / 'm.tdm'
    xml = ("<?xml version='1.0'?><root><localcolumns><localcolumn>"
           "<name>X</name><measurement_quantity>n</measurement_quantity>"
           "<sequence_representation>float</sequence_representation></localcolumn>"
           "</localcolumns></root>")
    meta.write_text(xml)
    monkeypatch.setenv('PYTHONPATH', str(Path.cwd()))
    # ensure fresh import to avoid runpy warning
    sys.modules.pop('indenter.load_datasets', None)
    monkeypatch.setattr(sys, 'argv', ['prog', str(data), str(meta)])
    runpy.run_module('indenter.load_datasets', run_name='__main__')
    out, err = capsys.readouterr()
    assert 'Text DataFrame' in out or 'Metadata DataFrame' in out


def test_cli_full(tmp_path, monkeypatch, capsys):
    """
    Running module with --full should print full TDM load message
    """
    data = tmp_path / 'd2.txt'
    data.write_text("T\n\nNumber of Points = 0\n\n")
    meta = tmp_path / 'm2.tdm'
    meta.write_text("")
    monkeypatch.setattr(mod.TdmsFile, 'read', lambda path: DummyTDMS())
     # ensure fresh import to avoid runpy warning
    sys.modules.pop('indenter.load_datasets', None)
    monkeypatch.setattr(sys, 'argv', ['prog', str(data), str(meta), '--full'])
    runpy.run_module('indenter.load_datasets', run_name='__main__')
    out, err = capsys.readouterr()
    assert 'Loaded full TDM' in out
