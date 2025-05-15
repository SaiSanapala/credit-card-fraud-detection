import streamlit as st
import pandas as pd

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

st.markdown("<h1 style='text-align: center;'> Credit Card Fraud Detection Dashboard – Built with XGBoost & Streamlit </h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Engineered by Sai Sanapala – End-to-End ML Fraud Detection System</p>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("View fraudulent transactions predicted by the model using this interactive dashboard.", unsafe_allow_html=True)

# Load the default fraud prediction dataset from kaggle
df = pd.read_csv("../data/flagged_frauds.csv")

# Show preview
st.subheader(" Preview of Flagged Transactions")
st.dataframe(df.head(100), use_container_width=True)

# Filtering slider
if 'Amount' in df.columns:
    st.sidebar.header("Filter Options")
    min_amt = float(df['Amount'].min())
    max_amt = float(df['Amount'].max())
    amount_threshold = st.sidebar.slider("Filter by transaction amount", min_amt, max_amt, min_amt)

    filtered_df = df[df['Amount'] > amount_threshold]

    st.markdown(f"### Showing {len(filtered_df)} suspicious transactions above **${amount_threshold:.2f}**")
    st.dataframe(filtered_df, use_container_width=True)

    # download
    st.download_button(
        label="Download filtered frauds as CSV",
        data=filtered_df.to_csv(index=False),
        file_name='filtered_frauds.csv',
        mime='text/csv'
    )
else:
    st.error("'Amount' column not found in dataset. Please recheck your data.")
