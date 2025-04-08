# test.py
import streamlit as st
import yfinance as yf

st.title("Test Streamlit App")

stock = st.text_input("Enter Stock Symbol", "GOOG")

if stock:
    st.write(f"Fetching data for: {stock}")
    data = yf.download(stock, '2012-01-01', '2022-12-31')
    st.write(data)
