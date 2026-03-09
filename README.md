# 🏦 Loan Approval Prediction — Complete Project

## 📁 Folder Structure
```
loan_project/
│
├── loan_approval_improved.ipynb   ← Run this first (training + saving model)
├── app.py                         ← Flask web app
├── requirements.txt               ← Python dependencies
├── LoanApprovalPrediction.csv     ← Add your dataset here!
├── model.pkl                      ← Auto-created after running notebook
├── columns.pkl                    ← Auto-created after running notebook
└── templates/
    └── index.html                 ← Web UI
```

---

## 🚀 Steps to Run

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Add your dataset
Place `LoanApprovalPrediction.csv` in this folder.

### Step 3 — Run the notebook
Open `loan_approval_improved.ipynb` and run ALL cells.
This will create `model.pkl` and `columns.pkl` at the end.

### Step 4 — Start Flask app
```bash
python app.py
```

### Step 5 — Open browser
Go to: **http://127.0.0.1:5000**

---

## ✅ What the notebook does
- Cleans & imputes missing data
- Engineers 7 new features (TotalIncome, EMI, ratios, log transforms)
- Trains 4 models: Logistic Regression, Random Forest, KNN, Gradient Boosting
- Tunes best model with GridSearchCV (Pipeline-based — no scaler mismatch!)
- Combines all into a Voting Ensemble
- Saves model.pkl + columns.pkl for Flask deployment

## ✅ What the Flask app does
- Takes applicant details via a web form
- Applies same feature engineering as training
- Returns Approved ✅ or Rejected ❌ with probability bar
