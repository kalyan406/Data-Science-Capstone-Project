import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Step 1: Load the dataset
df = pd.read_csv('/CAR_DETAILS.csv')

# Step 2: Inspect the data
print(df.head())  # Display first few rows
print(df.info())  # Summary of the dataset

# Step 3: Handle missing values (if any)
print(df.isnull().sum())  # Check for missing values

# Example of handling missing values: Drop rows with any missing values
df.dropna(inplace=True)

# Step 4: Data transformation - Encode categorical variables
# Example: One-hot encoding for categorical variables
df_encoded = pd.get_dummies(df, columns=['fuel', 'seller_type', 'transmission', 'owner'])

# Step 5: Data normalization/scaling (if necessary)
# Example: Scale numeric features using StandardScaler
scaler = StandardScaler()
numeric_features = ['year', 'selling_price', 'km_driven']
df_encoded[numeric_features] = scaler.fit_transform(df_encoded[numeric_features])

# Step 6: Data visualization (optional)
# Example: Visualize distributions or relationships using matplotlib or seaborn

# Step 7: Prepare data for modeling
# Example: Split data into training and testing sets
X = df_encoded.drop(['selling_price'], axis=1)
y = df_encoded['selling_price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Optional: Save processed data to a new CSV file if needed
df_encoded.to_csv('processed_car_details.csv', index=False)

# Optional: Further analysis or modeling using the processed data
