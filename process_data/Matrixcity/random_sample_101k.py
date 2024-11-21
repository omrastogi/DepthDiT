# %% Import necessary libraries
import pandas as pd
import os
import plotly.express as px


SAMPLE_DATA = False

# Define the base path
assert 'BASE_DIR' in os.environ, "Error: 'BASE_DIR' is not set in the environment variables."
base_path = os.getenv("BASE_DIR")
assert os.path.exists(base_path), f"Error: The directory {base_path} does not exist."

print("Libraries imported and base path defined.")


# %% Define helper functions

def convert_to_relative_paths(df, column_name, base_path):
    """Converts absolute paths to relative paths."""
    df[column_name] = df[column_name].apply(lambda x: os.path.relpath(x, base_path))
    return df

def extract_features(df, column_name):
    """Extracts Location Type, Data Type, Base File, and Tar ID from the Tar File path."""
    df['Location Type'] = df[column_name].str.extract(r'/(aerial|street)/')
    df['Data Type'] = df[column_name].str.extract(r'/(test|train)/')
    df['Base File'] = df[column_name].str.extract(r'([^/]+)\.tar$')[0]
    df['Tar ID'] = df['Location Type'] + '-' + df['Data Type'] + '-' + df['Base File']
    return df

def sample_tar_id(df, tar_id, n_samples):
    """Samples a specific number of rows for a given Tar ID."""
    available_samples = df[df["Tar ID"] == tar_id]
    return available_samples.sample(
        n=min(n_samples, len(available_samples)), random_state=42, replace=False
    )

def create_histogram(df, data_type, column='Tar ID'):
    """Creates and displays a histogram of value counts for Tar IDs."""
    counts = df[df['Data Type'] == data_type][column].value_counts().reset_index()
    counts.columns = [column, 'Count']
    fig = px.bar(counts, x=column, y='Count', title=f'Histogram of {column} Value Counts ({data_type.capitalize()} Set)')
    fig.update_xaxes(tickangle=90)
    fig.show()

print("Helper functions defined.")

# %% Load the CSV file and process data

# Load the CSV file
csv_file_path = "csv/tar_file_contents.csv"
tar_file_contents = pd.read_csv(csv_file_path)
# Convert absolute paths to relative paths
tar_file_contents = convert_to_relative_paths(tar_file_contents, "Tar File", base_path)
# Extract features from the Tar File column
tar_file_contents = extract_features(tar_file_contents, "Tar File")
# Save the updated CSV file
output_csv_path = "csv/Matrixcity_big_city_image_content_rel.csv"
tar_file_contents.to_csv(output_csv_path, index=False)
print(f"Updated CSV file saved successfully: {output_csv_path}")

# %% Visualize data with histograms
# TODO Uncomment if required
# # Create histogram for the train set
# create_histogram(tar_file_contents, data_type='train')
# # Create histogram for the test set
# create_histogram(tar_file_contents, data_type='test')
# print("Histograms generated.")


# %% Define sample proportions for train and test sets

street_train_samples = {
    "street-train-top_area": 33804,
    "street-train-right_area": 20853,
    "street-train-left_area": 20799,
    "street-train-bottom_area": 14544,
}

aerial_train_samples = {
    "aerial-train-big_high_block_1": 4012,
    "aerial-train-big_high_block_3": 1947,
    "aerial-train-big_high_block_2": 1676,
    "aerial-train-big_high_block_4": 1268,
    "aerial-train-big_high_block_5": 857,
    "aerial-train-big_high_block_6": 240,
}

street_test_samples = {
    "street-test-top_area_test": 280,
    "street-test-left_area_test": 240,
    "street-test-right_area_test": 160,
    "street-test-bottom_area_test": 120,
}

aerial_test_samples = {
    "aerial-test-big_high_block_1_test": 80,
    "aerial-test-big_high_block_3_test": 30,
    "aerial-test-big_high_block_2_test": 30,
    "aerial-test-big_high_block_4_test": 20,
    "aerial-test-big_high_block_5_test": 20,
    "aerial-test-big_high_block_6_test": 20,
}

print("Sample proportions defined.")

# %% Sample data based on defined proportions

# Initialize empty dataframes for each set
street_train_df = pd.DataFrame()
aerial_train_df = pd.DataFrame()
street_test_df = pd.DataFrame()
aerial_test_df = pd.DataFrame()

# Sample for street-train Tar IDs
for tar_id, sample_count in street_train_samples.items():
    sampled_df = sample_tar_id(tar_file_contents, tar_id, sample_count)
    street_train_df = pd.concat([street_train_df, sampled_df], ignore_index=True)

# Sample for aerial-train Tar IDs
for tar_id, sample_count in aerial_train_samples.items():
    sampled_df = sample_tar_id(tar_file_contents, tar_id, sample_count)
    aerial_train_df = pd.concat([aerial_train_df, sampled_df], ignore_index=True)

# Sample for street-test Tar IDs
for tar_id, sample_count in street_test_samples.items():
    sampled_df = sample_tar_id(tar_file_contents, tar_id, sample_count)
    street_test_df = pd.concat([street_test_df, sampled_df], ignore_index=True)

# Sample for aerial-test Tar IDs
for tar_id, sample_count in aerial_test_samples.items():
    sampled_df = sample_tar_id(tar_file_contents, tar_id, sample_count)
    aerial_test_df = pd.concat([aerial_test_df, sampled_df], ignore_index=True)

print("Data sampling completed.")

print(f"Number of samples in street_train_df: {len(street_train_df)}")
print(f"Number of samples in street_test_df: {len(street_test_df)}")
print(f"Number of samples in aerial_train_df: {len(aerial_train_df)}")
print(f"Number of samples in aerial_test_df: {len(aerial_test_df)}")


# %% Combine the sampled datasets
# Combine all the sampled datasets (street and aerial) into one
balanced_df = pd.concat([street_train_df, aerial_train_df, street_test_df, aerial_test_df], ignore_index=True)
# Report the total number of samples
print(f"Total samples in the dataset: {len(balanced_df)}")
# Save the combined dataset
output_path = "csv/sampled_data.csv"
if SAMPLE_DATA:
    balanced_df.to_csv(output_path, index=False)
else:
    tar_file_contents.to_csv(output_path, index=False)

print(f"Combined dataset saved successfully: {output_path}")
# %%
