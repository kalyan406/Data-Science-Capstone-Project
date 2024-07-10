#Saving the Best Model:
import joblib

# Assuming `best_model` is your trained model object

# Save the model to disk
filename = 'best_model.pkl'
joblib.dump(best_model, filename)
Loading the Model:
import joblib

# Load the model from disk
filename = 'best_model.pkl'
loaded_model = joblib.load(filename)
Example Integration with Data Preparation and Model Training:
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import joblib

# Step 1: Load the dataset
file_path = '/CAR_DETAILS.csv'  # Adjust the file path accordingly
df = pd.read_csv(file_path)

# Step 2: Handle missing values (if any)
df.dropna(inplace=True)  # Drop rows with missing values

# Step 3: Feature selection and encoding
categorical_features = ['fuel', 'seller_type', 'transmission', 'owner']
numeric_features = ['year', 'km_driven']  # Assuming these are numerical features

# Preprocessing pipeline
categorical_transformer = OneHotEncoder(handle_unknown='ignore')
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Step 4: Splitting data into training and testing sets
X = df.drop(['selling_price'], axis=1)
y = df['selling_price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Model training
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor())  # Example of using RandomForestRegressor
])

model.fit(X_train, y_train)

# Step 6: Evaluating the model (optional)
# Evaluation metrics, cross-validation, etc.

# Step 7: Save the best model
filename = 'best_model.pkl'
joblib.dump(model, filename)

# Step 8: Load the model
loaded_model = joblib.load(filename)

# Now `loaded_model` can be used for predictions or further analysis
