import flask
from flask import Flask, render_template, request
import pickle
import numpy as np
import sklearn

app = Flask(__name__)

model = pickle.load(open('rdf.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/getdata', methods=['POST'])
def pred():
    loanid = request.form['Loan ID']
    print(loanid)
    gender = request.form['Gender']
    print(loanid, gender)
    married = request.form['Married']
    print(loanid, gender, married)
    dependents = request.form['Dependents']
    print(loanid, gender, married, dependents)
    education = request.form['Education']
    print(loanid, gender, married, dependents, education)
    self_employed = request.form['Self_Employed']
    print(loanid, gender, married, dependents, education, self_employed)
    applicantincome = request.form['ApplicantIncome']
    print(loanid, gender, married, dependents, education, self_employed, applicantincome)
    coapplicantincome = request.form['CoapplicantIncome']
    print(loanid, gender, married, dependents, education, self_employed, applicantincome,
          coapplicantincome)
    loanamount = request.form['LoanAmount']
    print(loanid, gender, married, dependents, education, self_employed, applicantincome,
          coapplicantincome, loanamount)
    loan_amount_term = request.form['Loan_Amount_Term']
    print(loanid, gender, married, dependents, education, self_employed, applicantincome,
          coapplicantincome, loanamount, loan_amount_term)
    credit_history = request.form['Credit_History']
    print(loanid, gender, married, dependents, education, self_employed, applicantincome,
          coapplicantincome, loanamount, loan_amount_term, credit_history)
    property_area = request.form['Property_Area']
    print(loanid, gender, married, dependents, education, self_employed, applicantincome,
          coapplicantincome, loanamount, loan_amount_term, credit_history,
          property_area)

    inp_features = [[np.log(float(loanid)), int(gender), int(married), np.log(float(dependents)), int(education),
                     int(self_employed),
                     int(applicantincome),
                     int(coapplicantincome), int(loanamount), int(loan_amount_term), int(credit_history),
                     np.log(float(property_area))]]
    print(inp_features)
    prediction = model.predict(inp_features)
    print(type(prediction))
    t = prediction[0]
    print(t)
    if t > 0.5:
        prediction_text = 'Eligible to loan, Loan will be sanctioned'
    else:
        prediction_text = 'Not eligible to loan'
    print(prediction_text)
    return render_template('prediction.html', prediction_results=prediction_text)


if __name__ == "__main__":
    app.run()
