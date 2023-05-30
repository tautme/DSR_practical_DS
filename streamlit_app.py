import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression


def load_pickles(model_pickle_path, label_encoder_pickle_path):
    with open(model_pickle_path, "rb") as model_pickle_opener:
        model = pickle.load(model_pickle_opener)

    with open(label_encoder_pickle_path, "rb") as label_encoder_opener:
        label_encoder_dict = pickle.load(label_encoder_opener)

    return model, label_encoder_dict


def pre_process_data(df, label_encoder_dict):
    df_out = df.copy()
    df_out.replace(" ", 0, inplace=True)
    df_out.loc[:, 'TotalCharges'] = pd.to_numeric(df_out.loc[:, 'TotalCharges'])

    if 'customerID' in df_out.columns:
        df_out.drop('customerID', axis=1, inplace=True)

    for column, le in label_encoder_dict.items():
        if column in df_out.columns:
            df_out.loc[:, column] = le.transform(df_out.loc[:, column])

    return df_out

def make_predictions(test_data):
    model_pickle_path = "./models/churn_prediction_model.pkl"
    label_encoder_pickle_path = "./models/churn_prediction_label_encoder.pkl"

    model, label_encoder_dict = load_pickles(model_pickle_path, label_encoder_pickle_path)

    data_processed = pre_process_data(test_data, label_encoder_dict)
    if 'Churn' in test_data.columns:
        data_processed = data_processed.drop('Churn', axis=1)
    prediction = model.predict(data_processed)
    return prediction

if __name__ == "__main__":
    st.title("Customer churn prediction")
    data = pd.read_csv("./data/holdout_data.csv")

    ## use st.selectbox to run and predict churn for a single customer
    # st.table(data)
    # st.text("Select Customer")

    # customer = st.selectbox(
    #     "Select customer", 
    #     data['customerID'])

    gender = st.selectbox(
        "Select customer's gender: ", 
        ["Female", "Male"])

    senior_citizen_input = st.selectbox(
        "Is customer senior citizen?", 
        ['No', 'Yes'])
    senior_citizen = 1 if senior_citizen_input == 'Yes' else 0
    # senior_citizen = senior_citizen_input

    partner_input = st.selectbox(
        "Is customer partnered?", 
        ['No', 'Yes'])
    partner = 1 if partner_input == 'Yes' else 0
    partner = partner_input

    dependents_input = st.selectbox(
        "Does customer have dependents?", 
        ['No', 'Yes'])
    dependents = 1 if dependents_input == 'Yes' else 0
    dependents = dependents_input

    tenure = st.slider(
        "How long has customer been with the company?", 
        min_value=1, max_value=72, value=25, step=1)
    
    phone_service_input = st.selectbox(
        "Does customer have phone service?", 
        ['No', 'Yes'])
    phone_service = 1 if phone_service_input == 'Yes' else 0
    phone_service = phone_service_input

    multiple_lines_input = st.selectbox(
        "Does customer have multiple lines?", 
        ['No', 'Yes', 'No phone service'])
    multiple_lines = 1 if multiple_lines_input == 'Yes' else 0
    multiple_lines = multiple_lines_input

    internet_service_input = st.selectbox(
        "What internet service does customer have?", 
        ['DSL', 'Fiber optic', 'No'])
    internet_service = 1 if internet_service_input == 'Yes' else 0
    internet_service = internet_service_input

    online_security_input = st.selectbox(
        "Does customer have online security?", 
        ['No', 'Yes', 'No internet service'])
    online_security = 1 if online_security_input == 'Yes' else 0
    online_security = online_security_input

    online_backup_input = st.selectbox(
        "Does customer have online backup?", 
        ['No', 'Yes', 'No internet service'])
    online_backup = 1 if online_backup_input == 'Yes' else 0
    online_backup = online_backup_input

    device_protection_input = st.selectbox(
        "Does customer have device protection?", 
        ['No', 'Yes', 'No internet service'])
    device_protection = 1 if device_protection_input == 'Yes' else 0
    device_protection = device_protection_input

    tech_support_input = st.selectbox(
        "Does customer have tech support?", 
        ['No', 'Yes', 'No internet service'])
    tech_support = 1 if tech_support_input == 'Yes' else 0
    tech_support = tech_support_input

    streaming_tv = st.selectbox(
        "Does customer have streaming TV?", 
        ['Yes', 'No', 'No internet service'])

    streaming_movies = st.selectbox(
        "Does customer have streaming movies?", 
        ['Yes', 'No', 'No internet service'])

    contract = st.selectbox(
        "What contract does customer have?", 
        ['Month-to-month', 'One year', 'Two year'])

    paperless_billing_input = st.selectbox(
        "Does customer have paperless billing?", 
        ['Yes', 'No'])
    paperless_billing = 1 if paperless_billing_input == 'Yes' else 0
    paperless_billing = paperless_billing_input

    payment_method = st.selectbox(
        "What payment method does customer use?", 
        ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])

    monthly_charges = st.slider(
        "What are customer's monthly charges?", 
        min_value=1, max_value=200, value=50, step=1)
    
    total_charges = st.slider(
        "What are customer's total charges?", 
        min_value=1, max_value=10000, value=2000, step=1)

        
    customer_dict = {
        'gender': gender,
        'SeniorCitizen': senior_citizen, 
        'Partner': partner, 
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }


    customer_data = pd.DataFrame(
        [customer_dict]
    )

    st.table(customer_data)
    
    if st.button("Predict Churn"):
        prediction = make_predictions(customer_data)[0]
        prediction_string = "will churn" if prediction == 1 else "will not churn"
        st.text(f"Customer prediction: {prediction}")








    # customer_data = pd.DataFrame.from_dict(customer_dict)
    # st.table(customer_data)
    
    # if st.button("Predict Churn"):
    #     prediction = make_predictions(customer_data)[0]
    #     prediction_string = "will churn" if prediction == 1 else "will not churn"
    #     st.text(f"Customer prediction: {prediction}")















    # visualise customer's data
    # st.table(data)

    # if st.button("Predict Churn"):
    #     prediction = make_predictions(data)
    #     st.text(f"Customer prediction: {prediction}")

















## first example
## to use
## run 'streamlit run streamlit_app.py' in terminal

# import matplotlib.pyplot as plt
# import seaborn as sns

# data = pd.read_csv('./data/training_data.csv', index_col=0)
# st.write("Telco Churn Data")
# st.dataframe(data)

# st.write("How many custromers in the dataset churned?")
# target_counts = data['Churn'].value_counts()
# st.bar_chart(target_counts)


# st.write("What is the distribution of tenure?")
# st.bar_chart(data['tenure'].value_counts())

# st.write("View tenure vs. monthly charges")

# st.write(boxplot of monthly charges vs. churn)
# st.write(sns.boxplot(x='tenure', y='MonthlyCharges', data=data))
# st.pyplot()



