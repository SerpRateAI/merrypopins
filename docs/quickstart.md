## Quickstart

### Importing merrypopins Modules

```python linenums="1"
from pathlib import Path
from merrypopins.load_datasets import load_txt, load_tdm
from merrypopins.preprocess import default_preprocess, remove_pre_min_load, rescale_data, finalise_contact_index
from merrypopins.locate import default_locate
from merrypopins.make_dataset import merrypopins_pipeline
from merrypopins.statistics import default_statistics, calculate_stress_strain, default_statistics_stress_strain
```

### Load Indentation Data and Metadata

```python linenums="1"
# 1) Load indentation data:
data_file = Path("data/experiment1.txt")
df = load_txt(data_file)
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
# The channel metadata is stored as multiple rows with their respective columns
print(df_tdm_meta_channels.head(50))
```

### Preprocess Data

#### Option 1: Use default pipeline

```python linenums="1"
# This applies:
# 1. Removes all rows before minimum Load
# 2. Detects contact point and shifts Depth so contact = 0
# 3. Removes Depth < 0 rows and adds a flag for the contact point

df_processed = default_preprocess(df)

print(df_processed.head())
print("Contact point index:", df_processed[df_processed["contact_point"]].index[0])
```

#### Option 2: Customize each step (with optional arguments)

```python linenums="1"
# Step 1: Remove initial noise based on minimum Load
df_clean = remove_pre_min_load(df, load_col="Load (µN)")

# Step 2: Automatically detect contact point and zero the depth
df_rescaled = rescale_data(
    df_clean,
    depth_col="Depth (nm)",
    load_col="Load (µN)",
    N_baseline=30,     # number of points for baseline noise estimation
    k=5.0,             # noise threshold multiplier
    window_length=7,   # Savitzky-Golay smoothing window (must be odd)
    polyorder=2        # Polynomial order for smoothing
)

# Step 3: Trim rows before contact and/or flag the point
df_final = finalise_contact_index(
    df_rescaled,
    depth_col="Depth (nm)",
    remove_pre_contact=True,       # remove rows where depth < 0
    add_flag_column=True,          # add a boolean column marking the contact point
    flag_column="contact_point"    # customize the column name if needed
)

print(df_final[df_final["contact_point"]])  # display contact row
print("Contact point index:", df_final[df_final["contact_point"]].index[0])
```
🧪 Tip
You can omit or modify any step depending on your data:

- Skip remove_pre_min_load() if your data is already clean.
- Set remove_pre_contact=False if you want to retain all data.
- Customize flag_column to integrate with your own schema.

### Locate Pop-in Events

#### Detect Pop-ins using Default Method

```python linenums="1"
# Detect pop-ins using all methods
results = default_locate(df_processed)
print(results[results.popin])
```

### Customize Detection Thresholds

```python linenums="1"
results_tuned = default_locate(
    df_processed,
    iforest_contamination=0.002,
    cnn_threshold_multiplier=4.0,
    fd_threshold=2.5,
    savgol_threshold=2.0
)
```

### Visualize Detections

```python linenums="1"
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plt.plot(results_tuned["Depth (nm)"], results_tuned["Load (µN)"], label="Preprocessed", alpha=0.4, color='orange')

colors = {
    "popin_iforest": 'red',
    "popin_cnn": 'purple',
    "popin_fd": 'darkorange',
    "popin_savgol": 'green'
}
markers = {
    "popin_iforest": '^',
    "popin_cnn": 'v',
    "popin_fd": 'x',
    "popin_savgol": 'D'
}

for method, color in colors.items():
    mdf = results_tuned[results_tuned[method]]
    plt.scatter(mdf["Depth (nm)"], mdf["Load (µN)"],
                c=color, label=method.replace("popin_", "").capitalize(),
                marker=markers[method], alpha=0.7)

confident = results_tuned[results_tuned["popin_confident"]]
plt.scatter(confident["Depth (nm)"], confident["Load (µN)"],
            edgecolors='k', facecolors='none', label="Majority Vote (2+)", s=100, linewidths=1.5)

plt.xlabel("Depth (nm)"); plt.ylabel("Load (µN)")
plt.title("Pop-in Detections by All Methods")
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()
```

### Run Full Pipeline with merrypopins_pipeline

This function runs the entire merrypopins workflow, from loading data to locating pop-ins and generating visualizations.

#### Define Input and Output Paths

```python linenums="1"
# Define the text file that will be processed and output directory that will contain the visualization
text_file = Path("datasets/6microntip_slowloading/grain9_6um_indent03_HL_QS_LC.txt")
output_dir = Path("visualisations/6microntip_slowloading/grain9_6um_indent03_HL_QS_LC")

# Make sure output directory exists
output_dir.mkdir(parents=True, exist_ok=True)
```

#### Run The merrypopins Pipeline

```python linenums="1"
df_pipeline = merrypopins_pipeline(
    text_file,
    save_plot_dir=output_dir,
    trim_margin=30
)
```

#### View Result DataFrame

```python linenums="1"
df_pipeline.head()
```

#### View Result Visualizations

```python linenums="1"
# The pipeline generates plot in the specified output directory for the provided text file.
from PIL import Image
import matplotlib.pyplot as plt

# Load all PNGs from output folder
image_paths = sorted(output_dir.glob("*.png"))

# Only proceed if there are images
if image_paths:
    img = Image.open(image_paths[0])
    plt.figure(figsize=(12, 6))
    plt.imshow(img)
    plt.title(image_paths[0].stem)
    plt.axis('off')
    plt.show()
else:
    print("No plots found in output folder.")
```

### Calculate Pop-in Statistics

#### Calculate Pop-in Statistics (Load-Depth)

```python linenums="1"
df_statistics = default_statistics(df_pipeline)

# View the computed statistics for each pop-in
print(df_statistics.head())

```

### Calculate Stress-Strain Statistics

#### Perform Stress-Strain Transformation and Statistics

```python linenums="1"
# Perform stress-strain transformation
df_stress_strain = calculate_stress_strain(df_statistics)

# Calculate stress-strain statistics
df_stress_strain_statistics = calculate_stress_strain_statistics(df_stress_strain)

# View the calculated stress-strain statistics
print(df_stress_strain_statistics.head())
```

### Full Statistics Pipeline

#### Perform Default Full Statistics Pipeline for Stress-Strain

```python linenums="1"
df_statistics_stress_strain = default_statistics_stress_strain(
    df_pipeline,
    popin_flag_column="popin",
    before_window=0.5,
    after_window=0.5,
    Reff_um=5.323,
    min_load_uN=2000,
    smooth_stress=True,
    stress_col="stress",
    strain_col="strain",
    time_col="Time (s)",
)

# View the final stress-strain statistics
print(df_statistics_stress_strain.head())
```
