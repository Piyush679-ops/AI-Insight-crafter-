
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer , AutoModelForSeq2SeqLM
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error


# Cache the model and tokenizer to avoid reloading
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    return tokenizer, model

tokenizer, model = load_model()

# File validation
def validate_file(file):
    try:
        data = pd.read_csv(file)
        return data
    except Exception as e:
        return str(e)

# Generate data summary
def generate_summary(data):
    summary = data.describe(include="all")
    missing_values = data.isnull().sum()
    return summary, missing_values

# Plot correlation heatmap
def plot_heatmap(data):
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    st.pyplot(plt)

# Predictive Modeling
def train_predictive_model(data):
    # Drop rows with missing target values
    data_cleaned = data.dropna()
    
    # Identify numerical and categorical columns
    numerical_cols = data_cleaned.select_dtypes(include=['number']).columns
    categorical_cols = data_cleaned.select_dtypes(exclude=['number']).columns
    
    # Encode categorical columns if present
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        data_cleaned[col] = le.fit_transform(data_cleaned[col])
        label_encoders[col] = le

    # Check if there are sufficient numeric columns
    if len(numerical_cols) < 2:
        st.warning("Not enough numerical columns for prediction.")
        return None
    
    # Split the data into features and target variable
    X = data_cleaned.dropna(axis=1)  # Features (all columns except the last one)
    y = data_cleaned[numerical_cols[-1]]  # Target (the last numerical column)
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a simple Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate the Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)
    
    return model, mse, X_test, y_test, y_pred

# Streamlit UI
st.title("AI Insight Crafter Tool")

uploaded_file = st.file_uploader("Upload a CSV File")

task = st.selectbox("Select the task:", ["Generate Summary", "Visualize Data Correlation", "Generate Predictive Insights"])

if st.button("Run Task"):
    if uploaded_file:
        # Validate the file
        file_content = validate_file(uploaded_file)
        if isinstance(file_content, str):
            st.error(f"Error reading file: {file_content}")
        else:
            # Handle Data Summary
            if task == "Generate Summary":
                summary, missing_values = generate_summary(file_content)
                st.write("**Data Summary:**")
                st.write(summary)
                st.write("**Missing Values:**")
                st.write(missing_values)
                csv = summary.to_csv(index=False).encode('utf-8')
                st.download_button("Download Summary CSV", data=csv, file_name="data_summary.csv")

            # Handle Data Visualization
            elif task == "Visualize Data Correlation":
                numeric_data = file_content.select_dtypes(include=["number"])
                if numeric_data.empty:
                    st.warning("No numerical data available for correlation visualization.")
                else:
                    st.write("**Correlation Heatmap:**")
                    plot_heatmap(numeric_data)

            # Handle Predictive Insights
            elif task == "Generate Predictive Insights":
                model, mse, X_test, y_test, y_pred = train_predictive_model(file_content)
                if model is not None:
                    st.write("**Predictive Insights:**")
                    st.write(f"Mean Squared Error of the Model: {mse:.2f}")
                    
                    # Display comparison of actual vs predicted values
                    predictions_df = pd.DataFrame({
                        'Actual Values': y_test,
                        'Predicted Values': y_pred
                    })
                    st.write(predictions_df)
                    st.write("**Prediction Performance:**")
                    st.line_chart(predictions_df)
                    
                    # Option to download the results
                    csv_results = predictions_df.to_csv(index=False).encode('utf-8')
                    st.download_button("Download Predictions CSV", data=csv_results, file_name="predictions.csv")
                
    else:
        st.warning("Please upload a file first.")



