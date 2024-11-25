#%%
import os
import pandas as pd
# Read the CSV file
csv_path = "csv/filtered_data.csv"
filtered_df = pd.read_csv(csv_path)
assert 'BASE_DIR' in os.environ, "Error: 'BASE_DIR' is not set in the environment variables."
base_dir = os.getenv("BASE_DIR")
assert os.path.exists(base_dir), f"Error: The directory {base_dir} does not exist."

#%%
import json
# Load DINOv2 features from .jsonl
dinov2_features_path = "checkpoint/dinov2_embeddings.jsonl"

# Read the .jsonl file line by line and collect into a list
dinov2_features = []
with open(dinov2_features_path, 'r') as f:
    for line in f:
        item = json.loads(line)
        for image_path, embedding in item.items():
            dinov2_features.append({'rgb_filepath': image_path, 'features': embedding})

#%%
# Convert the list of dictionaries to a DataFrame
dinov2_df = pd.DataFrame(dinov2_features)
# Merge DINOv2 features with the filtered DataFrame
filtered_df = filtered_df.merge(dinov2_df, on='rgb_filepath', how='left')
# Rename the 'features' column for clarity
filtered_df.rename(columns={'features': 'rgb_dinov2_features'}, inplace=True)

# %%
import numpy as np
def extract_and_combine_features(filtered_df):
    rgb_features = filtered_df['rgb_dinov2_features']
    # Extract the arrays from the lists
    rgb_features_array = rgb_features.apply(lambda x: x[0])
    rgb_features_df = pd.DataFrame(rgb_features_array.tolist())
    rgb_features_df.columns = [f'rgb_feature_{i}' for i in range(rgb_features_df.shape[1])]
    rgb_features_df.reset_index(drop=True, inplace=True)
    return rgb_features_df

# %%
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import random

import pandas as pd
from sklearn.preprocessing import StandardScaler


# Step 1: Split the data into street and aerial subsets, then into train and test sets
df_street_train = filtered_df[(filtered_df["Location Type"] == "street") & (filtered_df["Data Type"] == "train")]
X_street_train = extract_and_combine_features(df_street_train)

df_street_test = filtered_df[(filtered_df["Location Type"] == "street") & (filtered_df["Data Type"] == "test")]
X_street_test = extract_and_combine_features(df_street_test)

df_aerial_train = filtered_df[(filtered_df["Location Type"] == "aerial") & (filtered_df["Data Type"] == "train")]
X_aerial_train = extract_and_combine_features(df_aerial_train)

df_aerial_test = filtered_df[(filtered_df["Location Type"] == "aerial") & (filtered_df["Data Type"] == "test")]
X_aerial_test = extract_and_combine_features(df_aerial_test)
# Step 2: Normalize the features
# For street data
scaler_street = StandardScaler()
X_street_train_scaled = scaler_street.fit_transform(X_street_train)
X_street_test_scaled = scaler_street.transform(X_street_test)

# For aerial data
scaler_aerial = StandardScaler()
X_aerial_train_scaled = scaler_aerial.fit_transform(X_aerial_train)
X_aerial_test_scaled = scaler_aerial.transform(X_aerial_test)

# Print the length of the DataFrames
print(f"Length of X_street_train: {len(X_street_train)}")
print(f"Length of X_street_test: {len(X_street_test)}")
print(f"Length of X_aerial_train: {len(X_aerial_train)}")
print(f"Length of X_aerial_test: {len(X_aerial_test)}")

# %%
# Apply K-Means clustering
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

def select_representative_images(data, features, k=5000, num_samples=20000, random_state=42, base_dir=None):
    """
    Apply K-Means clustering and select representative images from each cluster.

    Parameters:
        data (pd.DataFrame): DataFrame containing the data to cluster.
        features (np.ndarray): Feature matrix (rows correspond to rows in `data`).
        k (int): Number of clusters for K-Means.
        num_samples (int): Total number of images to select.
        random_state (int): Random state for K-Means initialization.

    Returns:
        pd.DataFrame: DataFrame containing the selected representative images.
    """
    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=random_state)
    data['Cluster'] = kmeans.fit_predict(features)

    centroids = kmeans.cluster_centers_
    distances = np.linalg.norm(features - centroids[data['Cluster']], axis=1)
    data['DistanceFromCentroid'] = distances

    # Calculate cluster sizes
    cluster_sizes = data['Cluster'].value_counts().sort_index()

    # Create cluster information DataFrame
    cluster_info = pd.DataFrame({
        'Cluster': cluster_sizes.index,
        'ClusterSize': cluster_sizes.values
    })

    # Proportional allocation of images to select
    total_images = cluster_info['ClusterSize'].sum()
    cluster_info['SelectionQuota'] = (cluster_info['ClusterSize'] / total_images) * num_samples
    cluster_info['ImagesToSelect'] = cluster_info['SelectionQuota'].round().astype(int)

    # Adjust for rounding errors
    total_selected = cluster_info['ImagesToSelect'].sum()
    difference = num_samples - total_selected
    if difference != 0:
        cluster_info['FractionalPart'] = cluster_info['SelectionQuota'] - cluster_info['ImagesToSelect']
        if difference > 0:
            cluster_info = cluster_info.sort_values(by='FractionalPart', ascending=False)
        else:
            cluster_info = cluster_info.sort_values(by='FractionalPart', ascending=True)
        for idx in cluster_info.index:
            if difference == 0:
                break
            cluster_info.at[idx, 'ImagesToSelect'] += np.sign(difference)
            difference -= np.sign(difference)
        cluster_info = cluster_info.drop(columns=['FractionalPart'])

    # Select images from each cluster
    selected_indices = []
    for idx, row in cluster_info.iterrows():
        cluster_id = row['Cluster']
        num_to_select = int(row['ImagesToSelect'])
        cluster_data = data[data['Cluster'] == cluster_id]

        # Sort by 'DistanceFromCentroid' descending
        cluster_data = cluster_data.sort_values(by='DistanceFromCentroid', ascending=False)

        # Select the required number of images
        selected = cluster_data.head(num_to_select)

        # Append indices
        selected_indices.extend(selected.index.tolist())

    # Final selected images
    selected_images = data.loc[selected_indices].copy()

    # Assemble the DataFrame in ascending order of Cluster
    selected_images = selected_images.sort_values(by='Cluster', ascending=True)
    selected_images = selected_images[['rgb_file_path', 'file_path']]
    selected_images = selected_images.rename(columns={"rgb_file_path": "rgb_file", "file_path": "depth_file"})
    if base_dir is not None: 
        selected_images['rgb_file'] = selected_images['rgb_file'].apply(lambda x: os.path.relpath(x, base_dir))
        selected_images['depth_file'] = selected_images['depth_file'].apply(lambda x: os.path.relpath(x, base_dir))

    return selected_images


# %%
final_data_street_train = select_representative_images(
    data=df_street_train,
    features=X_street_train_scaled,
    k=10000,
    num_samples=20000,
    random_state=42,
    base_dir=base_dir
)
print("Selected representative images for street train data.")
final_data_street_train.to_csv('csv/final_data_street_train_20k.csv', index=False)

# %%
final_data_street_test = select_representative_images(
    data=df_street_test,
    features=X_street_test_scaled,
    k=500,
    num_samples=1000,
    random_state=42,
    base_dir=base_dir
)
print("Selected representative images for street test data.")
final_data_street_test.to_csv('csv/final_data_street_test_1k.csv', index=False)

# %%
final_data_aerial_train = select_representative_images(
    data=df_aerial_train,
    features=X_aerial_train_scaled,
    k=5000,
    num_samples=5000,
    random_state=42,
    base_dir=base_dir
)
print("Selected representative images for aerial train data.")
final_data_aerial_train.to_csv('csv/final_data_aerial_train_5k.csv', index=False)

# %%
final_data_aerial_test = select_representative_images(
    data=df_aerial_test,
    features=X_aerial_test_scaled,
    k=500,
    num_samples=500,
    random_state=42,
    base_dir=base_dir
)
print("Selected representative images for aerial test data.")
final_data_aerial_test.to_csv('csv/final_data_aerial_test_0.5k.csv', index=False)

