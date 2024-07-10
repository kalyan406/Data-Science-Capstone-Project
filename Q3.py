import pandas as pd
# Load the dataset
data =pd.read_csv('/CAR_DETAILS.csv')

# Handling the null values
print(data.isnull().sum())

# Perform One-Hot Encoding
data_encoded = pd.get_dummies(data, columns=['fuel', 'seller_type', 'transmission', 'owner'], drop_first=True)

# Example of imputing missing values (if needed)
# data['some_column'].fillna(data['some_column'].mean(), inplace=True)

from sklearn.preprocessing import StandardScaler
# Example of scaling numerical data
scaler = StandardScaler()
numerical_cols = ['year', 'selling_price', 'km_driven']
data_encoded[numerical_cols] = scaler.fit_transform(data_encoded[numerical_cols])

# Save the preprocessed dataset to a new CSV file
data_encoded.to_csv('preprocessed_car_data.csv', index=False)
