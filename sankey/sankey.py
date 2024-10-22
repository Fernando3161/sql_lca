import pandas as pd
import plotly.graph_objects as go
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import warnings
import matplotlib
import os

from sankey_utils import (
    calculate_level,
    create_source,
    create_target,
    extract_flows,
    shorten_labels,
)

# Ignore specific warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)


# File path and sheet details
file_path = "pilot2.xlsx"  # Specify the path to your Excel file
sheet_name = "Upstream tree"  # Specify the sheet name

# Sankey Diagram Title
impact_name = "GWP"  # Global Warming Potential
unit_name = "kg_CO2"  # Unit of measurement

# Options for data processing
REMOVE_BACK_INFO = True  # Option to remove background information
SHORTEN_LABELS = True  # Option to shorten labels

# Read the Excel file into a DataFrame
df = pd.read_excel(
    os.path.join(os.getcwd(), "sankey", file_path), sheet_name=sheet_name, skiprows=1
)

# Rename the Result column to a standard name
res_name = [col for col in df.columns if "Result" in col][0]
df.rename(columns={res_name: "Result"}, inplace=True)

# Drop the 'Direct contribution' column if it exists
direc_cont = [col for col in df.columns if "Direct contribution" in col][0]
df.drop(direc_cont, axis=1, inplace=True)

# Filter DataFrame to keep only positive results
df = df[df["Result"] > 0]

# Drop columns that are completely empty
df.dropna(axis=1, how="all", inplace=True)

# Reset the index of the DataFrame
df.reset_index(drop=True, inplace=True)


# Fill NaN values with "X"
df = df.fillna("X")


# Add a 'Level' column to the DataFrame if it does not exist
if "Level" not in df.columns:
    df["Level"] = df.apply(calculate_level, axis=1)


# Create the Source and Target columns in the DataFrame
df["Source"] = create_source(df)
df["Target"] = create_target(df)


# Apply the extract_flows function to each row and create a new 'Flow' column in the DataFrame
df["Flow"] = df.apply(extract_flows, axis=1)

# Create a DataFrame containing unique flow names
unique_flows = df["Flow"].unique()
df_flows = pd.DataFrame(unique_flows, columns=["Flow"])  # Add a column name for clarity


# Map Source and Target indices to their corresponding flow names
df["Target Name"] = df["Target"].apply(
    lambda x: df["Flow"].iloc[int(x)] if pd.notna(x) else None
)
df["Source Name"] = df["Source"].apply(
    lambda x: df["Flow"].iloc[int(x)] if pd.notna(x) else None
)

# Map Source and Target names to their corresponding indices in the unique flows DataFrame
df["Source ID"] = df["Source Name"].apply(
    lambda x: df_flows[df_flows["Flow"] == x].index[0] if pd.notna(x) else None
)
df["Target ID"] = df["Target Name"].apply(
    lambda x: df_flows[df_flows["Flow"] == x].index[0] if pd.notna(x) else None
)

# Create the final DataFrame with relevant columns
df2 = df[["Result", "Source Name", "Target Name"]]


# This removes background information to avoid overflow of info.
if REMOVE_BACK_INFO:
    # Step 1: Identify all nodes that are directly linked to "market for" nodes
    df2["delete"] = 0  # Initialize the column to mark nodes for deletion
    # Create a set of unique source names linked to "market for" targets
    node_to_delete = set(
        df2[df2["Target Name"].str.startswith("market for", na=False)][
            "Source Name"
        ].unique()
    )

    # Step 2: Iteratively find all nodes linked to already marked nodes
    prev_node_count = -1  # To track changes in the node_to_delete set

    while (
        len(node_to_delete) > prev_node_count
    ):  # Keep iterating until no new nodes are added
        prev_node_count = len(node_to_delete)  # Update the previous count
        # Find all rows where the Target Name is in the set of nodes to delete
        newly_marked = df2[df2["Target Name"].isin(node_to_delete)][
            "Source Name"
        ].unique()
        node_to_delete.update(newly_marked)  # Add newly found nodes to delete set

    # Step 3: Mark the rows in the DataFrame for deletion
    df2["delete"] = df2["Source Name"].apply(lambda x: 1 if x in node_to_delete else 0)
    df2 = df2[df2["delete"] == 0]  # Keep only rows not marked for deletion

if SHORTEN_LABELS:
    # Remove the "market for " prefix from both 'Source Name' and 'Target Name' columns
    df2["Source Name"] = df2["Source Name"].str.replace(r"^market for ", "", regex=True)
    df2["Target Name"] = df2["Target Name"].str.replace(r"^market for ", "", regex=True)

    # Remove the '|' character and everything that follows it in both 'Source Name' and 'Target Name' columns
    df2["Source Name"] = df2["Source Name"].str.split("|").str[0]
    df2["Target Name"] = df2["Target Name"].str.split("|").str[0]

# Calculate the maximum value of the 'Result' column
max_value = df2["Result"].max()

# Normalize the 'Result' column relative to the maximum value
df2["Result"] = (df2["Result"] / max_value) * 100  # Normalize to a scale of 100


# Prepare the data for the Sankey diagram
sources = df2["Source Name"].tolist()
targets = df2["Target Name"].tolist()
values = df2["Result"].tolist()

# Remove entries where 'Target Name' is None since they don't have a target
sources_cleaned = []
targets_cleaned = []
values_cleaned = []
for source, target, value in zip(sources, targets, values):
    if target is not None:
        sources_cleaned.append(source)
        targets_cleaned.append(target)
        values_cleaned.append(value)

# Create a list of unique labels
labels = list(set(sources_cleaned + targets_cleaned))


# Shorten labels
if SHORTEN_LABELS:
    n_max = 20
else:
    n_max = 500
shortened_labels = shorten_labels(labels, max_length=n_max)

# Create a mapping of original labels to their shortened versions
label_mapping = {label: shortened for label, shortened in zip(labels, shortened_labels)}

# Map source and target names to their corresponding index in the shortened labels list
source_indices = [
    shortened_labels.index(label_mapping[source]) for source in sources_cleaned
]
target_indices = [
    shortened_labels.index(label_mapping[target]) for target in targets_cleaned
]

# Calculate node sizes based on outgoing flow values
node_sizes = np.zeros(len(shortened_labels))

# Aggregate values based on source indices
for source in source_indices:
    node_sizes[source] += values_cleaned[source_indices.index(source)]

# Ensure the last node size is 100
last_node_index = len(shortened_labels) - 1
last_node_size = node_sizes[last_node_index]

if last_node_size > 0:  # To avoid division by zero
    scaling_factor = 100 / last_node_size
    node_sizes *= scaling_factor

# Normalize the flow values for color mapping
normalized_values = (values_cleaned - np.min(values_cleaned)) / (
    np.max(values_cleaned) - np.min(values_cleaned)
)

# Create a colormap from Matplotlib for the flows
flow_cmap = cm.get_cmap("viridis", 256)  # Create a colormap with 256 colors
flow_colors = [
    mcolors.to_hex(flow_cmap(0.1 + val * 0.7)) for val in normalized_values
]  # Flow colors

# Normalize node sizes for blue color scale
normalized_node_sizes = (node_sizes - 0) / (np.max(node_sizes) - 0)

# Create a colormap from Matplotlib for the nodes
node_cmap = cm.get_cmap("Blues", 256)  # Create a colormap with 256 colors
node_colors = [
    mcolors.to_hex(node_cmap(0.5 + val * 0.5)) for val in normalized_node_sizes
]  # Node colors

# Create the Sankey diagram using Plotly
fig = go.Figure(
    go.Sankey(
        orientation="v",
        node=dict(
            pad=50,  # Increase the pad value for more horizontal spacing
            thickness=20,
            line=dict(color="black", width=0.5),
            label=shortened_labels,  # Use display labels for the plot
            color=node_colors,  # Blue scale for nodes
            hoverlabel=dict(
                bgcolor="rgba(0,0,0,0)"
            ),  # Set label background to transparent
        ),
        link=dict(
            source=source_indices,  # Indices of the source nodes
            target=target_indices,  # Indices of the target nodes
            value=values_cleaned,  # Flow values
            color=flow_colors,  # Color from Matplotlib colormap for flows
        ),
    )
)

# Update layout to set the height of the figure
fig.update_layout(
    title_text=f"Sankey Diagram | {impact_name} | Max Value: {max_value:.2f} {unit_name}",
    font_size=10,
    height=800,  # Adjust height as needed
    width=800,
)

# Add hovertemplate to the links to show full names
fig.update_traces(
    link=dict(
        hovertemplate="From: %{source.label}<br>To: %{target.label}<br>Value: %{value}"
    )
)

# Save the figure as an HTML file
fig.write_html(
    os.path.join(os.getcwd(), "sankey", f"sankey_diagram_{impact_name}.html")
)
