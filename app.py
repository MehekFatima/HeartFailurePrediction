import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open("classifier.pkl", 'rb'))

# Mapping for categorical features
sex_map = {'Male': 0, 'Female': 1}
chest_pain_type_map = {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-anginal Pain': 2, 'Asymptomatic': 3}
fasting_bs_map = {'Lower than 120 mg/dl': 0, 'Greater than 120 mg/dl': 1}
resting_ecg_map = {'Normal': 0, 'ST-T Wave Abnormality': 1, 'Left Ventricular Hypertrophy': 2}
exercise_angina_map = {'No': 0, 'Yes': 1}
st_slope_map = {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2}

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    age = float(request.form['age'])
    sex = sex_map[request.form['sex']]
    chest_pain_type = chest_pain_type_map[request.form['chest_pain_type']]
    resting_bp = float(request.form['resting_bp'])
    cholesterol = float(request.form['cholesterol'])
    fasting_bs = fasting_bs_map[request.form['fasting_bs']]
    resting_ecg = resting_ecg_map[request.form['resting_ecg']]
    max_hr = float(request.form['max_hr'])
    exercise_angina = exercise_angina_map[request.form['exercise_angina']]
    oldpeak = float(request.form['oldpeak'])
    st_slope = st_slope_map[request.form['st_slope']]

    input_features = [age, sex, chest_pain_type, resting_bp, cholesterol, fasting_bs,
                      resting_ecg, max_hr, exercise_angina, oldpeak, st_slope]

    df = pd.DataFrame([input_features], columns=["Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS",
                                                 "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope"])

    output = model.predict(df)

    if output == 1:
        res_val = "** heart disease **"
    else:
        res_val = "no heart disease "

    return render_template('index.html', prediction_text='Patient has {}'.format(res_val))


if __name__ == "__main__":
    app.run()






