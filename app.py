from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load('income_model.pkl')
le = joblib.load('label_encoder.pkl')

FIELDS = [
    'age', 'workclass', 'fnlwgt', 'education', 'educational-num',
    'marital-status', 'occupation', 'relationship', 'race',
    'gender', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country'
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = {field: request.form[field] for field in FIELDS}
    df = pd.DataFrame([data])

    numeric = ['age', 'fnlwgt', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    for col in numeric:
        df[col] = df[col].astype(int)

    prediction = model.predict(df)[0]
    income = le.inverse_transform([prediction])[0]

    return jsonify({'income': income})

if __name__ == '__main__':
    app.run(debug=True)
