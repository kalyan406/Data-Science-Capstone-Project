import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Step 1: Load the dataset
df = pd.read_csv('/CAR_DETAILS.csv')

# Step 2: Handle missing values (if any)
df.dropna(inplace=True)  # Drop rows with missing values

# Step 3: Feature selection (example)
# Assuming relevant features are 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner'
X = df[['year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner']]
y = df['selling_price']

# Step 4: Feature encoding (One-hot encoding for categorical variables)
# Create a column transformer for encoding categorical features
categorical_features = ['fuel', 'seller_type', 'transmission', 'owner']
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features)
    ])

# Step 5: Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Data scaling (if necessary, example using StandardScaler)
# Example: Scale numerical features
numeric_features = ['year', 'km_driven']
numeric_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Step 7: Create a pipeline with preprocessing and the model (example)
# For demonstration purposes, let's use a simple Linear Regression model
from sklearn.linear_model import LinearRegression

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Step 8: Fit the model on the training data
model.fit(X_train, y_train)

# Step 9: Evaluate the model on the test data
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_pred = model.predict(X_test)

print("Metrics:")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
print(f"R-squared: {r2_score(y_test, y_pred)}")
