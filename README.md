💻 Laptop Price Prediction using Machine Learning  

## 📌 Overview  
This project aims to predict laptop prices based on key specifications such as **RAM, GPU, screen resolution, processor type, and weight**. Using a **Random Forest Regressor**, the model provides accurate predictions after preprocessing and cleaning the dataset.  

## 🚀 Features  
- **Data Preprocessing:** Handles missing values and formats dataset properly.  
- **Machine Learning Model:** Implements Random Forest for price prediction.  
- **Visualization & Analysis:** Uses Matplotlib & Seaborn for insights.  
- **Power BI Dashboard (Optional):** Showcases trends in laptop pricing.  
- **Web Application (Optional):** Streamlit-based UI for predictions.  

## 📂 Dataset  
The dataset contains **1300+ laptops**, including details such as:  
✔ **Company** (Apple, Dell, HP, Lenovo, etc.)  
✔ **Type** (Ultrabook, Notebook, Gaming)  
✔ **RAM, GPU, ScreenResolution, CPU**  
✔ **Weight** (kg)  
✔ **Operating System**  
✔ **Price** (target variable)  

## 🔧 Installation  
### Step 1️⃣: Clone the Repository  
```bash
git clone https://github.com/your-username/Laptop-Price-Prediction.git
cd Laptop-Price-Prediction

Step 2️⃣: Install Dependencies
bash
pip install -r requirements.txt

Step 3️⃣: Run the Model
bash
python laptop_price_prediction.py
🛠 Model Training Pipeline
The project follows a structured ML pipeline: 
1️⃣ Data Cleaning & Preprocessing 
2️⃣ Feature Engineering (RAM, GPU, Weight, etc.)
3️⃣ Splitting Data (Train-Test Split) 
4️⃣ Model Training (Random Forest Regressor) 
5️⃣ Evaluation (R² Score, Mean Squared Error) 
6️⃣ Visualization (Feature Importance, Prediction Accuracy)

📊 Visualization & Insights
🔹 Feature Importance (Impact on Price)
python
import matplotlib.pyplot as plt
import seaborn as sns

# Extract feature importance from model
importances = model_pipeline.named_steps["randomforestregressor"].feature_importances_
feature_names = X_train.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=feature_names)
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.title("Feature Importance in Laptop Price Prediction")
plt.show()
🔹 Error Analysis (Actual vs. Predicted Prices)
python
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs. Predicted Laptop Prices")
plt.show()

📊 Power BI Dashboard Integration
To showcase your results interactively:
1️⃣ Export Predictions as CSV
python
results.to_csv("laptop_price_predictions.csv", index=False)
2️⃣ Import into Power BI → Create Scatter Plot
3️⃣ Add Insights & Filters for better visualization

🖥 Web Application (Optional)
🔹 Streamlit UI for Easy Predictions
python
import streamlit as st
import pandas as pd
import joblib  # Load trained model

# Load model
model = joblib.load("laptop_price_model.pkl")

st.title("Laptop Price Prediction")
st.write("Enter laptop specifications to estimate the price.")

# User inputs
company = st.selectbox("Company", ["Apple", "HP", "Dell", "Lenovo"])
ram = st.slider("RAM (GB)", 4, 64, 8)
weight = st.slider("Weight (kg)", 1.0, 5.0, 2.0)

# Prediction
input_data = pd.DataFrame([[company, ram, weight]], columns=["Company", "Ram", "Weight"])
predicted_price = model.predict(input_data)[0]

st.write(f"Estimated Price: **₹{predicted_price:.2f}**")
Run with:
bash
streamlit run app.py

