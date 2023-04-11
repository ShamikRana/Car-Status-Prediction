from flask import Flask, request, render_template
import pickle
import json
import numpy as np

app = Flask(__name__)

with open("model.pickle", "rb") as f:
    model = pickle.load(f)

with open("columns.json", "r") as f:
    columns = json.load(f)["data_columns"]

def predict_status(Model, Year, Price):
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
    
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods = ["POST"])
def predict():
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