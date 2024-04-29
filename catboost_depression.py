from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd

catboost_df = pd.read_csv("catboost.csv")

# Converting categorical columns to the appropriate type
catboost_df['LocationDesc'] = catboost_df['LocationDesc'].astype('category')
catboost_df['Stratification1'] = catboost_df['Stratification1'].astype('category')

# Split into features and target
features = catboost_df[['Stratification1', 'LocationDesc', 'YearStart']]
target = catboost_df['DataValue']

# Splitting into training and test sets, with a 80-20 ratio for training and testing
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize the CatBoost model with default parameters
catboost_model = CatBoostRegressor(verbose=True, cat_features=["LocationDesc", "Stratification1", "YearStart"])

# Train the model on the training set
catboost_model.fit(X_train, y_train)
catboost_model.save_model('catboost_model.cbm', format='cbm')

# Predict on the test set to evaluate the model's performance
y_pred = catboost_model.predict(X_test)

# Calculate the Mean Squared Error (MSE) as an evaluation metric
mse = mean_squared_error(y_test, y_pred)

print(mse)  # Return the Mean Squared Error as a measure of model accuracy
