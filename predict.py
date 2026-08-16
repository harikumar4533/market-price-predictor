import joblib
import pandas as pd

model = joblib.load("model.pkl")

def predict_price(open_price, high_price, low_price, volume):
    data = pd.DataFrame([{
        "Open": open_price,
        "High": high_price,
        "Low": low_price,
        "Volume": volume
    }])

    prediction = model.predict(data)
    return prediction[0]

if __name__ == "__main__":
    price = predict_price(140, 145, 137, 27000)
    print("Predicted Close Price:", price)
