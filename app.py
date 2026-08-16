import streamlit as st
from predict import predict_price

st.set_page_config(
    page_title="Market Price Predictor",
    page_icon="📈"
)

st.title("📈 Market Price Predictor")
st.write("Predict a closing price using a simple Machine Learning model.")

open_price = st.number_input("Open Price", value=140.0)
high_price = st.number_input("High Price", value=145.0)
low_price = st.number_input("Low Price", value=137.0)
volume = st.number_input("Volume", value=27000)

if st.button("Predict Price"):
    prediction = predict_price(
        open_price,
        high_price,
        low_price,
        volume
    )

    st.success(f"Predicted Close Price: {prediction:.2f}")
