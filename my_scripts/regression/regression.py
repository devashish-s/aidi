import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler


# Load the combined CSV file
data_path = '/media/nithin/17b8fbe0-a0a3-46d6-bb0e-d8bebe7bcbd0/experimental/KeypointsRepos/detectron2/training_datasets/datasets/src/Female_augmented_combined_output_with_height.csv'
df = pd.read_csv(data_path)
output_model_path = '/media/nithin/17b8fbe0-a0a3-46d6-bb0e-d8bebe7bcbd0/experimental/KeypointsRepos/detectron2/Female_regression_model_height_augmented.pkl'


############################ NORMALIZATION ###########################
# # Selecting columns to normalize (excluding the 'Person Name')
# columns_to_normalize = df.columns.difference(['Person Name'])
#
# # Applying Min-Max Scaling
# scaler = MinMaxScaler()
# df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

##############################################################################


# Select input features and target variables
X = df[["front_shoulder_dist","front_chest_dist","front_waist_dist","front_hip_dist","front_height_in_pixels_dist",
        "side_chest_dist","side_waist_dist","side_hip_dist","side_height_in_pixels_dist", 'Height(inch)']]
y = df[['Shoulder(inch)', 'Chest(inch)', 'Waist(inch)', 'Hip(inch)']]

# X = df[['front_pose_shoulder_dist', 'front_pose_chest_dist', 'front_pose_waist_dist', 'front_pose_hip_dist', 'front_pose_height',
#         'side_pose_chest_dist', 'side_pose_waist_dist','side_pose_hip_dist' , 'side_pose_height', 'Height(inch)']]
# y = df[['Shoulder(inch)', 'Chest(inch)', 'Waist(inch)', 'Hip(inch)']]

# Verify the number of rows
print(f"Total number of rows: {len(df)}")

# First, split out 10 rows for testing
X_train_valid, X_test, y_train_valid, y_test, names_train_valid, names_test = train_test_split(X, y, df['person_name'], test_size=0.1, random_state=42)

# Then, split the remaining rows into 180 for training and the rest for validation
X_train, X_val, y_train, y_val = train_test_split(X_train_valid, y_train_valid, train_size=0.9, random_state=42)

# Verify the sizes of the splits
print(f"Training set size: {len(X_train)}")
print(f"Validation set size: {len(X_val)}")
print(f"Testing set size: {len(X_test)}")



# Initialize and train the regression model on the training data
model = LinearRegression()
model.fit(X_train, y_train)

# Validate the model on the validation set
y_val_pred = model.predict(X_val)
val_mse = mean_squared_error(y_val, y_val_pred)
print(f"Validation Mean Squared Error: {val_mse}")

val_mse = mean_absolute_error(y_val, y_val_pred)
print(f"Validation mean_absolute_error: {val_mse}")


# Evaluate the model on the test set
y_test_pred = model.predict(X_test)
test_mse = mean_squared_error(y_test, y_test_pred)
print(f"Test Mean Squared Error: {test_mse}")

test_mse = mean_absolute_error(y_test, y_test_pred)
print(f"Test mean_absolute_error: {test_mse}")

import joblib

# Save the trained model to a file
joblib.dump(model, output_model_path)
print(f"Model saved to {output_model_path}")


results_df = pd.DataFrame({
    'Person Name': names_test,
    'Actual Shoulder(inch)': y_test['Shoulder(inch)'],
    'Predicted Shoulder(inch)': y_test_pred[:, 0],
    'Actual Chest(inch)': y_test['Chest(inch)'],
    'Predicted Chest(inch)': y_test_pred[:, 1],
    'Actual Waist(inch)': y_test['Waist(inch)'],
    'Predicted Waist(inch)': y_test_pred[:, 2],
    'Actual Hip(inch)': y_test['Hip(inch)'],
    'Predicted Hip(inch)': y_test_pred[:, 3]
})

# Save the DataFrame to a CSV file
results_path = '/media/nithin/17b8fbe0-a0a3-46d6-bb0e-d8bebe7bcbd0/experimental/KeypointsRepos/detectron2/regression_results_female.csv'
results_df.to_csv(results_path, index=False)
print(f"Results saved to {results_path}")


