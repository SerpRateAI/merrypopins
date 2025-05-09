# Quickstart

```python
from pathlib import Path
from indenter.load_datasets import load_txt, load_tdm

# 1) Load indentation data:
data_file = Path("data/experiment1.txt")
df = load_txt(data_file)

# The indentation data is stored as multiple rows with their respective columns
print(df.head())
print("Timestamp:", df.attrs['timestamp'])
print("Number of Points:", df.attrs['num_points'])

# 2) Load tdm metadata:
tdm_meta_file = Path("data/experiment1.tdm")

# Load tdm metadata and channels this will create dataframe for root and channels
df_tdm_meta_root, df_tdm_meta_channels = load_tdm(tdm_meta_file)

# The root metadata is stored as one row with their respective columns
print(df_tdm_meta_root.head())

# To be able to read all the columns of root metadata dataframe it can be transposed
df_tdm_meta_root = df_tdm_meta_root.T.reset_index()
df_tdm_meta_root.columns = ['attribute', 'value']
print(df_tdm_meta_root.head(50))

# The chanel metadata is stored as multiple rows with their respective columns
print(df_tdm_meta_channels.head(50))
```