#%%
import pandas as pd
# Read the CSV file
csv_path = "csv/filtered_data.csv"
filtered_df = pd.read_csv(csv_path)

#%%
import json
# Load DINOv2 features from .jsonl
dinov2_features_path = "checkpoint/dinov2_embeddings.jsonl"

# Read the .jsonl file line by line and collect into a list
dinov2_features = []
with open(dinov2_features_path, 'r') as f:
    for line in f:
        item = json.loads(line)
        # Each item is a dictionary with a single key-value pair
        # Extract the key (image path) and value (embedding list)
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

filtered_street = filtered_df[filtered_df["Location Type"]=="street"]
filtered_aerial = filtered_df[filtered_df["Location Type"]=="aerial"]

X_street = extract_and_combine_features(filtered_df[filtered_df["Location Type"]=="street"])
X_aerial = extract_and_combine_features(filtered_df[filtered_df["Location Type"]=="aerial"])

# Normalize the features
scaler = StandardScaler()
X_street_scaled = scaler.fit_transform(X_street)
X_aerial_scaled = scaler.fit_transform(X_aerial)

# %%
# Apply K-Means clustering
k = 5000
num_samples = 20000
kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
filtered_street['Cluster'] = kmeans.fit_predict(X_street)

centroids = kmeans.cluster_centers_
distances = np.linalg.norm(X_street - centroids[filtered_street['Cluster']], axis=1)
filtered_street['DistanceFromCentroid'] = distances

#%%
# Calculate cluster sizes
cluster_sizes = filtered_street['Cluster'].value_counts().sort_index()

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
print(difference)
if difference != 0:
    cluster_info['FractionalPart'] = cluster_info['SelectionQuota'] - cluster_info['ImagesToSelect']
    # For positive difference, add to clusters with highest fractional part
    # For negative difference, subtract from clusters with lowest fractional part
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

# Verify the total images selected
total_selected = cluster_info['ImagesToSelect'].sum()
print("Total images to select after adjustment:", total_selected)

# Select images from each cluster
selected_indices = []

for idx, row in cluster_info.iterrows():
    cluster_id = row['Cluster']
    num_to_select = int(row['ImagesToSelect'])
    cluster_data = filtered_street[filtered_street['Cluster'] == cluster_id]

    # Sort by 'DistanceFromCentroid' descending
    cluster_data = cluster_data.sort_values(by='DistanceFromCentroid', ascending=False)

    # Select the required number of images
    selected = cluster_data.head(num_to_select)

    # Append indices
    selected_indices.extend(selected.index.tolist())

# Final selected images
selected_images = filtered_street.loc[selected_indices].copy()
# Save the final selected images DataFrame as final_data2
final_data2 = selected_images.copy()
# Assemble the DataFrame in ascending order of Cluster
final_data2 = final_data2.sort_values(by='Cluster', ascending=True)

final_data2.to_csv('csv/final_data.csv', index=False)

print("Total images selected:", len(selected_images))