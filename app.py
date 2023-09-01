from flask import Flask, request, render_template
import pickle
import json
import numpy as np

app = Flask(__name__)

# Load the model and the columns
with open("model.pickle", "rb") as f:
    model = pickle.load(f)

# Load the columns
with open("columns.json", "r") as f:
    columns = json.load(f)["data_columns"]

# Function to predict the status of the car
def predict_status(Model: str, Year: int, Price: float) -> str:
    index = np.where(np.array(columns) == Model)[0]
    x = np.zeros(len(columns))
    x[0] = Year
    x[1] = Price
    x[index] = 1
    status = model.predict([x])[0]
    if status == 0:
        return "Old"
    if status == 1:
        return "New"
    
# Home page
@app.route("/")
def home() -> str:
    return render_template("home.html")

# Predict page
@app.route("/predict", methods = ["POST"])
def predict() -> str:
    model = request.form["model"]
    year = int(request.form["year"])
    price = float(request.form["price"])

    result = predict_status(model, year, price)
    
    if model.lower() in columns:
        return render_template("predict.html", result = result)
    else:
        return render_template("error.html")
    
if __name__=="__main__":
    app.run(debug=True)