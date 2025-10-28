import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib
import warnings

# ignore those  warnings
warnings.filterwarnings('ignore')

print("Model training Script...")

# 1. Load All Datasets
try:
    # keep_default_na=False is super important here.
    # We have "None" as a string value, and we don't want pandas to turn it into a NaN.
    orders = pd.read_csv("orders.csv", keep_default_na=False)
    routes = pd.read_csv("routes_distance.csv", keep_default_na=False)
    
    # These other files are fine, no "None" string issues
    performance = pd.read_csv("delivery_performance.csv")
    inventory = pd.read_csv("warehouse_inventory.csv")
    
    print("...all CSVs loaded successfully.")

except FileNotFoundError as e:
    print(f"Whoops, file not found: {e}.")
    print("Make sure all CSV files (orders, routes, performance, inventory) are in the same folder.")
    exit()

# 2. Data Cleaning & Renaming
# Applying the quick fixes we figured out in the notebook

# 'None' is a valid string
orders['Special_Handling'] = orders['Special_Handling'].replace('None', 'Normal Handling')
routes['Weather_Impact'] = routes['Weather_Impact'].replace('None', 'Normal Weather')

orders['Order_Date'] = pd.to_datetime(orders['Order_Date'])

print("...data cleaned up.")

# 3. Data Merging

# orders and performance
full_data = pd.merge(orders, performance, on="Order_ID", how="inner")

# Add the route info
full_data = pd.merge(full_data, routes, on="Order_ID", how="inner")

# Add the warehouse info.
# Need to rename the 'Location' col in the inventory table first so it matches 'Origin'
inventory.rename(columns={'Location': 'Origin'}, inplace=True)
full_data = pd.merge(full_data, inventory, on=["Origin", "Product_Category"], how="left")

print(f"...data merged, got {len(full_data)} total records to work with.")

# 4. Feature Engineering
# Create our target variable: delivery late or not?
full_data['is_delayed'] = (full_data['Actual_Delivery_Days'] > full_data['Promised_Delivery_Days']).astype(int)

# Let's fill NaNs with 0, assuming no data means 0 stock or reorder level.
full_data['Current_Stock_Units'] = full_data['Current_Stock_Units'].fillna(0)
full_data['Reorder_Level'] = full_data['Reorder_Level'].fillna(0)
full_data['Storage_Cost_per_Unit'] = full_data['Storage_Cost_per_Unit'].fillna(0)

# This was our new engineered feature: is the item below reorder level?
full_data['Is_Below_Reorder'] = (full_data['Current_Stock_Units'] < full_data['Reorder_Level']).astype(int)

print("...feature engineering done.")

# 5. Define Features (X) and Target (y) for the model
categorical_features = [
    'Priority', 
    'Carrier', 
    'Special_Handling', 
    'Weather_Impact', 
    'Origin', 
    'Destination',
    'Customer_Segment',
    'Product_Category'
]

# the numerical ones
numerical_features = [
    'Distance_KM', 
    'Traffic_Delay_Minutes',
    'Order_Value_INR',
    'Is_Below_Reorder' # Our new feature!
]

# X = our features
X = full_data[categorical_features + numerical_features]
# y = what we want to predict
y = full_data['is_delayed']

print(f"Features selected: {categorical_features + numerical_features}")

# 6. Create Preprocessing & Model Pipeline

# Set up the column transformer.
# This will OneHotEncode the categorical stuff and just let the numerical stuff pass through.
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', 'passthrough', numerical_features)
    ])

# Define the model
classifier = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)

# Bundle the preprocessor and the classifier into one handy pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', classifier)
])

print("...model pipeline created.")

# 7. Split and Train Model
# Standard train/test split. Stratify on 'y' to keep the delay proportion same in both sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.")

# train
model_pipeline.fit(X_train, y_train)

print("...model training complete.")

# 8. Evaluate Model

y_pred = model_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy on Test Set: {accuracy * 100:.2f}%")

# Save the whole pipeline (preprocessor + model) so we can use it later
save_path = 'delivery_optimizer_model.joblib'
joblib.dump(model_pipeline, save_path)

print(f"Model pipeline saved to '{save_path}'")