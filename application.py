# Importing necessary libraries
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Loading the dataset
df = pd.read_csv('Churning_project.csv')

# Data Cleaning (as done in the previous steps)
df = df.drop_duplicates()
df['Email'] = df['Email'].fillna('Unknown')
df['Address'] = df['Address'].fillna('Unknown')
df['Avatar'] = df['Avatar'].fillna('Unknown')
df['Avg. Session Length'] = df['Avg. Session Length'].fillna(df['Avg. Session Length'].mean())
df['Time on App'] = df['Time on App'].fillna(df['Time on App'].mean())
df['Time on Website'] = df['Time on Website'].fillna(df['Time on Website'].mean())
df['Length of Membership'] = df['Length of Membership'].fillna(df['Length of Membership'].mean())
df['Yearly Amount Spent'] = df['Yearly Amount Spent'].fillna(df['Yearly Amount Spent'].mean())

# Creating the target variable and feature set
df['Churn'] = df['Yearly Amount Spent'].apply(lambda x: 1 if x < 500 else 0)
X = df[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership', 'Yearly Amount Spent']]
y = df['Churn']

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Training the model
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# Streamlit App
st.title("Customer Churn Prediction App")
st.write("This app predicts whether a customer is likely to churn based on their data.")

# User input fields
st.header("Input Customer Data")
avg_session_length = st.number_input("Average Session Length", value=30.0)
time_on_app = st.number_input("Time on App", value=10.0)
time_on_website = st.number_input("Time on Website", value=20.0)
length_of_membership = st.number_input("Length of Membership (years)", value=3.0)
yearly_amount_spent = st.number_input("Yearly Amount Spent", value=500.0)

# Prediction function
def predict_churn(avg_session_length, time_on_app, time_on_website, length_of_membership, yearly_amount_spent):
    input_data = pd.DataFrame([[avg_session_length, time_on_app, time_on_website, length_of_membership, yearly_amount_spent]], 
                              columns=['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership', 'Yearly Amount Spent'])
    prediction = model.predict(input_data)
    return "Churn" if prediction[0] == 1 else "No Churn"

# Predict button
if st.button("Predict Churn"):
    result = predict_churn(avg_session_length, time_on_app, time_on_website, length_of_membership, yearly_amount_spent)
    st.write("Prediction:", result)

# Display model metrics (optional)
if st.checkbox("Show Model Performance on Test Set"):
    y_pred = model.predict(X_test)
    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    st.write("Precision:", precision_score(y_test, y_pred))
    st.write("Recall:", recall_score(y_test, y_pred))
    st.write("F1 Score:", f1_score(y_test, y_pred))

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    st.write("Confusion Matrix:")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)