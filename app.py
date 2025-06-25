import gradio as gr
import pandas as pd
import pickle

best_model = pickle.load(open('xgboost_model.pkl', 'rb'))
label_encoders = pickle.load(open('label_encoders.pkl', 'rb'))
feature_columns = pickle.load(open('feature_columns.pkl', 'rb'))

# Prediction function
def predict_churn(gender, senior, partner, dependents, tenure, phone, multiple, internet,
                  online_sec, online_back, device, tech, stream_tv, stream_mov,
                  contract, paperless, payment, monthly_charges, total_charges):
    
    input_data = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": int(senior),
        "Partner": partner,
        "Dependents": dependents,
        "tenure": int(tenure),
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": online_sec,
        "OnlineBackup": online_back,
        "DeviceProtection": device,
        "TechSupport": tech,
        "StreamingTV": stream_tv,
        "StreamingMovies": stream_mov,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": float(monthly_charges),
        "TotalCharges": float(total_charges)
    }])
    
    for col, le in label_encoders.items():
        if col in input_data.columns:
            input_data[col] = le.transform(input_data[col])

    input_data = input_data[feature_columns]
    
    prob = best_model.predict_proba(input_data)[:, 1][0]
    churn = best_model.predict(input_data)[0]
    churn_label = label_encoders['Churn'].inverse_transform([churn])[0]

    return f"Churn Probability: {prob:.2f} | Prediction: {churn_label}"

# Gradio Interface
iface = gr.Interface(
    fn=predict_churn,
    inputs=[
        gr.Dropdown(["Female", "Male"], label="Gender"),
        gr.Checkbox(label="Senior Citizen (Check if Yes)"),
        gr.Dropdown(["Yes", "No"], label="Partner"),
        gr.Dropdown(["Yes", "No"], label="Dependents"),
        gr.Number(label="Tenure (months)", precision=0),
        gr.Dropdown(["Yes", "No"], label="Phone Service"),
        gr.Dropdown(["Yes", "No", "No phone service"], label="Multiple Lines"),
        gr.Dropdown(["DSL", "Fiber optic", "No"], label="Internet Service"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Online Security"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Online Backup"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Device Protection"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Tech Support"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Streaming TV"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Streaming Movies"),
        gr.Dropdown(["Month-to-month", "One year", "Two year"], label="Contract"),
        gr.Dropdown(["Yes", "No"], label="Paperless Billing"),
        gr.Dropdown(["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], label="Payment Method"),
        gr.Number(label="Monthly Charges"),
        gr.Number(label="Total Charges")
    ],
    outputs="text",
    title="Customer Churn Prediction"
)

iface.launch()

