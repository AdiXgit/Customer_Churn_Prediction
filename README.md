# Customer Churn Prediction 📉💰

## Overview
This project predicts **customer churn** using an **XGBoost model** trained on customer data. The model takes inputs such as tenure, monthly charges, total charges, contract type, and payment method to determine whether a customer is likely to leave the service.

## How Churn Prediction Works 🧠
Churn prediction models like **XGBoost** work by identifying patterns in customer behavior. The model analyzes past customer data and learns **what kind of customers tend to leave (churn) and what kind tend to stay**. 

### 🏗 How does the model make predictions?
1. **It looks at customer features** – The model considers:
   - How long the customer has been with the company (**Tenure**)
   - How much they pay monthly (**Monthly Charges**)
   - Their total spending so far (**Total Charges**)
   - What type of contract they have (**Month-to-Month, One Year, Two Year**)
   - How they make payments (**Credit Card, Bank Transfer, etc.**)
   
2. **It compares the input to past churn patterns** – The model has learned from past customers who left and those who stayed.

3. **It assigns a probability** – The model outputs a number between 0 and 1. 
   - If **close to 1 (e.g., 0.75 or 75%)**, the customer is likely to churn.
   - If **close to 0 (e.g., 0.10 or 10%)**, the customer is likely to stay.

4. **Final Decision:**
   - If the probability is **greater than 50%**, the model **flags the customer as a high churn risk**.
   - If it's lower, the customer is **expected to stay**.

## Model Used ⚙️
- RandomForest
- **XGBoost (Extreme Gradient Boosting)**: A powerful machine learning algorithm that builds multiple decision trees and combines them to make highly accurate predictions.
- **Why XGBoost?**
  - It handles large datasets efficiently.
  - It finds complex relationships between customer behavior and churn.
  - It prevents overfitting with regularization techniques.

## Running the Project 🚀
### 1️⃣ Install Dependencies
Make sure you have Python installed, then install the required packages:

```bash
pip install streamlit pandas numpy joblib xgboost
```
