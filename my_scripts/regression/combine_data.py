import pandas as pd
import os

# Paths to the CSV files
front_distances_csv_path = 'training_datasets/datasets/Female Customers_data/front_distances.csv'
side_distances_csv_path = 'training_datasets/datasets/Female Customers_data/side_distances.csv'
summary_csv_path = 'training_datasets/datasets/src/actual_values.csv'
output_csv_path = 'training_datasets/datasets/src/Female_augmented_combined_output_with_height.csv'


# Load the CSV files
front_df = pd.read_csv(front_distances_csv_path)
side_df = pd.read_csv(side_distances_csv_path)
actual_values_df = pd.read_csv(summary_csv_path)

# Rename the column in actual_values_df to match the other dataframes
actual_values_df.rename(columns={"Person Name": "person_name"}, inplace=True)

# Merge the dataframes on 'person_name' with an inner join
merged_df = front_df.merge(side_df, on='person_name', how='inner').merge(actual_values_df, on='person_name', how='inner')

# Move the 'person_name' column to the last position
cols = list(merged_df.columns)
cols.append(cols.pop(cols.index('person_name')))
merged_df = merged_df[cols]

# Optionally save to a new CSV file if needed
merged_df.to_csv(output_csv_path, index=False)
print("Merge complete and CSV saved.")
