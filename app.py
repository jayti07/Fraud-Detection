import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title='Credit Card Fraud Detection', layout='wide')

# Fix pandas styler limit
pd.set_option("styler.render.max_elements", 600000)

# Load model and scaler
model = joblib.load('fraud_model.pkl')
scaler = joblib.load('scaler.pkl')

# Title
st.title('💳 Credit Card Fraud Detection')
st.write('Upload a CSV file to detect fraudulent transactions')

uploaded_file = st.file_uploader('Upload CSV', type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Save original amount before scaling
    original_amount = df['Amount'].copy()

    # Preprocess
    if 'Time' in df.columns:
        df = df.drop('Time', axis=1)
    if 'Class' in df.columns:
        df = df.drop('Class', axis=1)

    df['Amount'] = scaler.transform(df[['Amount']])

    # Predict
    predictions = model.predict(df)

    # Build results with ORIGINAL amount
    result_df = pd.DataFrame({
        'Transaction ID': range(1, len(predictions) + 1),
        'Amount ($)': original_amount.values,
        'Prediction': ['Fraud' if p == 1 else 'Legitimate' for p in predictions]
    })

    fraud_count = (predictions == 1).sum()
    legit_count = (predictions == 0).sum()
    total = len(predictions)

    # --- Summary Cards ---
    st.write('### Summary')
    col1, col2, col3 = st.columns(3)
    col1.metric('Total Transactions', total)
    col2.metric('Legitimate', legit_count)
    col3.metric('Fraudulent 🚨', fraud_count)

    st.markdown('---')

    # --- Pie Chart + Histogram ---
    col4, col5 = st.columns(2)

    with col4:
        st.write('### Fraud vs Legitimate')
        fig1, ax1 = plt.subplots()
        ax1.pie(
            [legit_count, fraud_count],
            labels=['Legitimate', 'Fraud'],
            colors=['#2ecc71', '#e74c3c'],
            autopct='%1.2f%%',
            startangle=90
        )
        ax1.axis('equal')
        st.pyplot(fig1)

    with col5:
        st.write('### Transaction Amount Distribution')
        fig2, ax2 = plt.subplots()
        ax2.hist(original_amount, bins=50, color='#3498db', edgecolor='black')
        ax2.set_xlabel('Amount ($)')
        ax2.set_ylabel('Count (Log Scale)')
        ax2.set_title('Amount Distribution')
        ax2.set_yscale('log')
        st.pyplot(fig2)

    st.markdown('---')

    # --- Fraud Transactions Table ---
    st.write('### Flagged Fraud Transactions')

    fraud_df = result_df[result_df['Prediction'] == 'Fraud'].copy()

    def highlight_fraud(row):
        if row['Prediction'] == 'Fraud':
            return ['background-color: #e74c3c; color: white'] * len(row)
        else:
            return ['background-color: #2ecc71; color: white'] * len(row)

    st.write(f"Showing {len(fraud_df)} flagged fraud transactions:")
    st.dataframe(
        fraud_df.style.apply(highlight_fraud, axis=1),
        use_container_width=True
    )

    st.markdown('---')

    # --- Download Button ---
    st.write('### Download Results')
    csv = result_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label='📥 Download Full Results as CSV',
        data=csv,
        file_name='fraud_detection_results.csv',
        mime='text/csv'
    )