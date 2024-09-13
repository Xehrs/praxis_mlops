import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

# Model Load
model = joblib.load('rental_price_model.joblib')

room_count  = int(input("Enter num of rooms: "))
area_sqft   = float(input("Enter area in sqft: "))

user_input  = np.array([[room_count,area_sqft]])

predict_rental_price    = model.predict(user_input)[0]

print(f"The predicted rental price for rooms count = {room_count} and area in sqft = {area_sqft} is = {predict_rental_price}")