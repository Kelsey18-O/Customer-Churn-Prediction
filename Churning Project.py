# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Step 1: Load the dataset
df = pd.read_csv('Churning_project.csv')

# Step 2: Initial data exploration
print("Initial Dataset Head:")
print(df.head())
print("\nData Info:")
print(df.info())
print("\nMissing Values Per Column:\n", df.isnull().sum())

# Step 3: Data Cleaning
# Removing duplicates
df = df.drop_duplicates()

# Filling missing values for categorical columns
df['Email'] = df['Email'].fillna('Unknown')
df['Address'] = df['Address'].fillna('Unknown')
df['Avatar'] = df['Avatar'].fillna('Unknown')

# Filling missing values for numerical columns with the mean
df['Avg. Session Length'] = df['Avg. Session Length'].fillna(df['Avg. Session Length'].mean())
df['Time on App'] = df['Time on App'].fillna(df['Time on App'].mean())
df['Time on Website'] = df['Time on Website'].fillna(df['Time on Website'].mean())
df['Length of Membership'] = df['Length of Membership'].fillna(df['Length of Membership'].mean())
df['Yearly Amount Spent'] = df['Yearly Amount Spent'].fillna(df['Yearly Amount Spent'].mean())

# Confirming that all missing values have been handled
print("\nMissing Values Per Column After Filling:\n", df.isnull().sum())

# Adding a 'Churn' column based on 'Yearly Amount Spent'
df['Churn'] = df['Yearly Amount Spent'].apply(lambda x: 1 if x < 500 else 0)

# Step 4: Define features (X) and target (y)
y = df['Churn']
X = df[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership', 'Yearly Amount Spent']]

# Train-test split with the updated feature set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 5: Train the model
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# Step 6: Make predictions on the test set
y_pred = model.predict(X_test)

# Step 7: Model evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Plotting the confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Step 8: Creating a function for prediction (to use in the Streamlit app)
def predict_churn(avg_session_length, time_on_app, time_on_website, length_of_membership, yearly_amount_spent):
    input_data = pd.DataFrame([[avg_session_length, time_on_app, time_on_website, length_of_membership, yearly_amount_spent]], 
                              columns=['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership', 'Yearly Amount Spent'])
    prediction = model.predict(input_data)
    return "Churn" if prediction[0] == 1 else "No Churn"

# Example usage of the prediction function (for testing)
print(predict_churn(34.5, 12.1, 40.3, 4.5, 500))


