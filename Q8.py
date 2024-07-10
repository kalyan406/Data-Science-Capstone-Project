import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Step 1: Load the original CAR DETAILS dataset
file_path = '/CAR_DETAILS.csv'  # Adjust the file path accordingly
df = pd.read_csv(file_path)

# Step 2: Randomly select 20 data points
random_data = df.sample(n=20, random_state=42)  # Randomly select 20 data points

# Step 3: Load the saved model
filename = 'best_model.pkl'
loaded_model = joblib.load(filename)

# Step 4: Prepare data for prediction (similar preprocessing steps as used during training)
# Assuming categorical and numerical features used during training are the same

categorical_features = ['fuel', 'seller_type', 'transmission', 'owner']
numeric_features = ['year', 'km_driven']

# Preprocessing pipeline
categorical_transformer = OneHotEncoder(handle_unknown='ignore')
numeric_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Step 5: Preprocess the randomly selected data
X_new = random_data.drop(['selling_price'], axis=1)
y_true = random_data['selling_price']

# Transform the features
X_new_transformed = preprocessor.fit_transform(X_new)

# Step 6: Make predictions using the loaded model
y_pred = loaded_model.predict(X_new_transformed)

# Step 7: Evaluate the model's performance (optional)
# You can compare y_pred with y_true to evaluate performance metrics

# Example: Print predicted and true values for comparison
results = pd.DataFrame({'True Selling Price': y_true, 'Predicted Selling Price': y_pred})
print(results)
