from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Pipeline includes scaler + model — no separate scaler.pkl needed
model   = pickle.load(open('model.pkl', 'rb'))
columns = pickle.load(open('columns.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        gender          = int(request.form['gender'])
        married         = int(request.form['married'])
        dependents      = int(request.form['dependents'])
        education       = int(request.form['education'])
        self_employed   = int(request.form['self_employed'])
        applicant_inc   = float(request.form['applicant_income'])
        coapplicant_inc = float(request.form['coapplicant_income'])
        loan_amount     = float(request.form['loan_amount'])
        loan_term       = float(request.form['loan_term'])
        credit_history  = float(request.form['credit_history'])
        property_area   = request.form['property_area']

        prop_semiurban = 1 if property_area == 'Semiurban' else 0
        prop_urban     = 1 if property_area == 'Urban'     else 0

        # Feature Engineering — same as training
        total_income      = applicant_inc + coapplicant_inc
        emi               = loan_amount / (loan_term if loan_term > 0 else 1)
        income_loan_ratio = total_income / (loan_amount + 1)
        remaining_income  = total_income - (emi * 1000)
        loan_amount_log   = np.log1p(loan_amount)
        total_income_log  = np.log1p(total_income)
        applicant_inc_log = np.log1p(applicant_inc)

        raw = {
            'Dependents':              dependents,
            'ApplicantIncome':         applicant_inc,
            'CoapplicantIncome':       coapplicant_inc,
            'LoanAmount':              loan_amount,
            'Loan_Amount_Term':        loan_term,
            'Credit_History':          credit_history,
            'TotalIncome':             total_income,
            'EMI':                     emi,
            'IncomeToLoanRatio':       income_loan_ratio,
            'RemainingIncome':         remaining_income,
            'LoanAmount_log':          loan_amount_log,
            'TotalIncome_log':         total_income_log,
            'ApplicantIncome_log':     applicant_inc_log,
            'Gender_Male':             gender,
            'Married_Yes':             married,
            'Education_Not Graduate':  education,
            'Self_Employed_Yes':       self_employed,
            'Property_Area_Semiurban': prop_semiurban,
            'Property_Area_Urban':     prop_urban,
        }

        # Build DataFrame with exact column order — prevents mismatch!
        features_df = pd.DataFrame(0, index=[0], columns=columns)
        for col, val in raw.items():
            if col in features_df.columns:
                features_df[col] = val

        # Pipeline handles scaling internally
        prediction  = model.predict(features_df)[0]
        probability = model.predict_proba(features_df)[0][1]

        result = {
            'approved':    bool(prediction == 1),
            'probability': round(float(probability) * 100, 1),
            'message':     'Loan Approved! ✅' if prediction == 1 else 'Loan Rejected ❌'
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
