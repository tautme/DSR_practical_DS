import streamlit as st
import pandas as pd
import pickle
# from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression


def load_pickles(model_pickle_path):
    with open(model_pickle_path, "rb") as model_pickle_opener:
        model = pickle.load(model_pickle_opener)

    return model

# def pre_process_data(df, label_encoder_dict):
#     df_out = df.copy()
#     df_out.replace(" ", 0, inplace=True)
#     df_out.loc[:, 'TotalCharges'] = pd.to_numeric(df_out.loc[:, 'TotalCharges'])

#     if 'customerID' in df_out.columns:
#         df_out.drop('customerID', axis=1, inplace=True)

#     for column, le in label_encoder_dict.items():
#         if column in df_out.columns:
#             df_out.loc[:, column] = le.transform(df_out.loc[:, column])

#     return df_out

def make_predictions(test_data):
    model_pickle_path = "./models/fraud_prediction_model.pkl"
    # label_encoder_pickle_path = "./models/churn_prediction_label_encoder.pkl"

    model = load_pickles(model_pickle_path)

    # data_processed = pre_process_data(test_data, label_encoder_dict)
    # if 'Churn' in test_data.columns:
    #     data_processed = data_processed.drop('Churn', axis=1)
    prediction = model.predict(customer_data)
    return prediction


#     if st.button("Predict Fraud"):
#         prediction = make_predictions(transaction_data)[0]
#         prediction_string = "Fraud" if prediction == 1 else "Not Fraud"
#         st.text(f"Fraud prediction: {prediction}")


################################





    # customer_data = pd.DataFrame.from_dict(customer_dict)
    # st.table(customer_data)
    
    # if st.button("Predict Churn"):
    #     prediction = make_predictions(customer_data)[0]
    #     prediction_string = "will churn" if prediction == 1 else "will not churn"
    #     st.text(f"Customer prediction: {prediction}")








################################

# data = pd.read_csv('./data/transaction_dataset.csv', index_col=0)
st.write("Etherium Transaction Data")






# if st.button("Predict Fraud"):
#     prediction = make_predictions(data)
#     st.text(f"Fraud prediction: {prediction}")

















## first example
## to use
## run 'streamlit run streamlit_app.py' in terminal

# import matplotlib.pyplot as plt
# import seaborn as sns

# data = pd.read_csv('./data/transaction_dataset.csv', index_col=0)
# st.write("Etherium Transaction Data")
# st.dataframe(data)

# st.write("How many transactions in the dataset fraud?")
# target_counts = data['FLAG'].value_counts()
# st.bar_chart(target_counts)


# st.write("What is the distribution of tenure?")
# st.bar_chart(data['tenure'].value_counts())

# st.write("View tenure vs. monthly charges")

# st.write(boxplot of monthly charges vs. churn)
# st.write(sns.boxplot(x='tenure', y='MonthlyCharges', data=data))
# st.pyplot()



if __name__ == "__main__":
    st.title("Transaction fraud prediction")

    ambst = st.slider(
        "What is the average time between sent transactions? (minutes)",
        min_value=1.0, max_value=4000.0, value=5.0, step=1.0)
    
    ambrt = st.slider(
        "What is the average time between received transactions? (minutes)",
        min_value=1.0, max_value=4800.0, value=5.0, step=1.0)
    
    tdbfl = st.slider(
        "What is the time difference between first and last transactions? (minutes)",
        min_value=1.0, max_value=100.0, value=5.0, step=1.0)
    
    sentt = st.slider(
        "How many transactions has the customer sent?",
        min_value=1.0, max_value=100.0, value=5.0, step=1.0)
    
    rect = st.slider(
        "How many transactions has the customer received?",
        min_value=1.0, max_value=100.0, value=5.0, step=1.0)
    
    ncc = st.slider(
        "How many contracts has the customer created?",
        min_value=1.0, max_value=100.0, value=5.0, step=1.0)
    
    urfa = st.slider(
        "How many unique addresses has the customer received from?",
        min_value=1.0, max_value=100.0, value=5.0, step=1.0)
    
    usta = st.slider(
        "How many unique addresses has the customer sent to?",
        min_value=1.0, max_value=100.0, value=5.0, step=1.0)
    
    minvr = st.slider(
        "What is the minimum value received?",
        min_value=1.0, max_value=100.0, value=5.0, step=1.0)
    
    maxvr = st.slider(
        "What is the maximum value received?",
        min_value=1.0, max_value=100.0, value=5.0, step=1.0)
    
    avgvr = st.slider(
        "What is the average value received?",
        min_value=1.0, max_value=100.0, value=5.0, step=1.0)
    
    minvs = st.slider(
        "What is the minimum value sent?",
        min_value=1.0, max_value=100.0, value=5.0, step=1.0)
    
    maxvs = st.slider(
        "What is the maximum value sent?",
        min_value=1.0, max_value=100.0, value=5.0, step=1.0)
    
    avgvs = st.slider(
        "What is the average value sent?",
        min_value=1.0, max_value=100.0, value=5.0, step=1.0)
    
    minvsc = st.slider(
        "What is the minimum value sent to contract?",
        min_value=1.0, max_value=100.0, value=5.0, step=1.0) 
    
    maxvsc = st.slider(
        "What is the maximum value sent to contract?",
        min_value=1.0, max_value=100.0, value=5.0, step=1.0)
    
    avgvsc = st.slider(
        "What is the average value sent to contract?",
        min_value=1.0, max_value=100.0, value=5.0, step=1.0)
    
    ttc = st.slider(
        "What is the total number of transactions?",
        min_value=1.0, max_value=100.0, value=5.0, step=1.0)

    ter = st.slider(
        "What is the total ether received?",
        min_value=1.0, max_value=100.0, value=5.0, step=1.0)
    
    tes = st.slider(
        "What is the total ether sent?",
        min_value=1.0, max_value=100.0, value=5.0, step=1.0)
    
    tesc = st.slider(
        "What is the total ether sent to contracts?",
        min_value=1.0, max_value=100.0, value=5.0, step=1.0)
    
    teb = st.slider(
        "What is the total ether balance?",
        min_value=1.0, max_value=1000.0, value=5.0, step=1.0)
    
transaction_dict = {
    'Avg min between sent tnx': ambst, 
    'Avg min between received tnx': ambrt,
    'Time Diff between first and last (Mins)': tdbfl, 
    'Sent tnx': sentt, 
    'Received Tnx': rect,
    'Number of Created Contracts': ncc, 
    'Unique Received From Addresses': urfa,
    'Unique Sent To Addresses': usta, 
    'min value received': minvr, 
    'max value received ': maxvr,
    'avg val received': avgvr, 
    'min val sent': minvs, 
    'max val sent': maxvs, 
    'avg val sent': avgvs,
    'min value sent to contract': minvsc, 
    'max val sent to contract': maxvsc,
    'avg value sent to contract': avgvsc,
    'total transactions (including tnx to create contract': ttc,
    'total Ether sent': tes, 
    'total ether received': ter,
    'total ether sent contracts': tesc, 
    'total ether balance': teb
    }


customer_data = pd.DataFrame(
    [transaction_dict]
)

#     st.table(customer_data)
    
if st.button("Predict Fraud"):
    prediction = make_predictions(customer_data)[0]
    prediction_string = "Fraud" if prediction == 1 else "Not Fraud"
    st.text(f"Prediction: {prediction_string}")

## create horizontal line here
import time
with st.spinner("Processing..."):
    time.sleep(3)
st.success("Done!")

data = pd.read_csv('./data/transaction_dataset.csv', index_col=0)
st.write("Etherium Transaction Data Used in Training: FLAG = 1 is Fraud")


st.dataframe(data)
st.write("Search for a transaction by hash address here: https://etherscan.io/")
st.write("Dataset from here: https://www.kaggle.com/datasets/vagifa/ethereum-frauddetection-dataset")