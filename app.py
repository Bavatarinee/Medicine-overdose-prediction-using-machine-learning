import pickle
import pandas as pd
from flask import Flask, render_template, request
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import os

app = Flask(__name__)

# Global variables to hold accuracy and confusion matrix
MODEL_ACC = None
MODEL_CM = None

# ✅ Categorize age into ranges
def categorize_age(age):
    if age <= 5:
        return "0-5"
    elif age <= 10:
        return "6-10"
    elif age <= 20:
        return "11-20"
    elif age <= 30:
        return "21-30"
    elif age <= 40:
        return "31-40"
    elif age <= 50:
        return "41-50"
    elif age <= 60:
        return "51-60"
    elif age <= 70:
        return "61-70"
    elif age <= 80:
        return "71-80"
    else:
        return "81-90"

# ✅ Load and prepare dataset
def load_and_prepare_dataset():
    df = pd.read_csv("medicine_overdose_data.csv")

    # Convert numeric ages in dataset to ranges
    df['Age'] = df['Age'].apply(lambda x: categorize_age(x) if isinstance(x, (int, float)) else x)

    # Fill missing values
    df[['Medicine 1', 'Medicine 2', 'Medicine 3']] = df[['Medicine 1', 'Medicine 2', 'Medicine 3']].fillna('None')
    df['Overdose Symptoms'] = df['Overdose Symptoms'].fillna('None')

    # Label Encoders
    le_med1 = LabelEncoder()
    le_med2 = LabelEncoder()
    le_med3 = LabelEncoder()
    le_symptoms = LabelEncoder()
    le_age = LabelEncoder()

    df['Medicine 1 Enc'] = le_med1.fit_transform(df['Medicine 1'])
    df['Medicine 2 Enc'] = le_med2.fit_transform(df['Medicine 2'])
    df['Medicine 3 Enc'] = le_med3.fit_transform(df['Medicine 3'])
    df['Symptoms Enc'] = le_symptoms.fit_transform(df['Overdose Symptoms'])
    df['Age Enc'] = le_age.fit_transform(df['Age'])

    X = df[['Medicine 1 Enc', 'Dose 1', 'Medicine 2 Enc', 'Dose 2',
            'Medicine 3 Enc', 'Dose 3', 'Age Enc', 'Symptoms Enc']]
    y = df['Outcome (Overdose)']

    medicine_list = sorted(set(df['Medicine 1']).union(df['Medicine 2']).union(df['Medicine 3']))

    return X, y, le_med1, le_med2, le_med3, le_symptoms, le_age, medicine_list

# ✅ Train and save model with SMOTE + Gradient Boosting
def train_model():
    global MODEL_ACC, MODEL_CM

    X, y, le1, le2, le3, le_symptoms, le_age, _ = load_and_prepare_dataset()

    # ⚡ Apply SMOTE to balance classes
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, stratify=y_res, random_state=42
    )

    # ✅ Tuned Gradient Boosting
    model = GradientBoostingClassifier(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    MODEL_ACC = round(accuracy_score(y_test, y_pred) * 100, 2)
    MODEL_CM = confusion_matrix(y_test, y_pred).tolist()

    print("✅ Model Accuracy:", MODEL_ACC)
    print("✅ Confusion Matrix:\n", MODEL_CM)

    # Save trained model
    with open('overdose_model_multi.pkl', 'wb') as f:
        pickle.dump((model, le1, le2, le3, le_symptoms, le_age), f)

# ✅ Load model
def load_model():
    with open('overdose_model_multi.pkl', 'rb') as f:
        return pickle.load(f)

@app.route('/')
def home():
    _, _, _, _, _, _, _, medicine_list = load_and_prepare_dataset()
    return render_template(
        'index.html',
        prediction=None,
        medicine_list=medicine_list,
        acc=MODEL_ACC,
        cm=MODEL_CM
    )

@app.route('/predict', methods=['POST'])
def predict():
    model, le1, le2, le3, le_symptoms, le_age = load_model()

    # Get inputs
    med1 = request.form['medicine1']
    med2 = request.form['medicine2']
    med3 = request.form['medicine3']
    dose1 = request.form['dose1']
    dose2 = request.form['dose2']
    dose3 = request.form['dose3']
    age_range = request.form['age']
    symptoms = request.form['symptoms']

    # Encode input
    med1_enc = le1.transform([med1])[0]
    med2_enc = le2.transform([med2])[0]
    med3_enc = le3.transform([med3])[0]
    symp_enc = le_symptoms.transform([symptoms])[0]
    age_enc = le_age.transform([age_range])[0]

    input_data = pd.DataFrame([[med1_enc, float(dose1), med2_enc, float(dose2),
                                med3_enc, float(dose3), age_enc, symp_enc]],
                              columns=['Medicine 1 Enc', 'Dose 1', 'Medicine 2 Enc', 'Dose 2',
                                       'Medicine 3 Enc', 'Dose 3', 'Age Enc', 'Symptoms Enc'])

    prediction = model.predict(input_data)[0]
    result = "Overdose" if prediction == 1 else "No Overdose"

    _, _, _, _, _, _, _, medicine_list = load_and_prepare_dataset()
    return render_template(
        'index.html',
        prediction=result,
        medicine_list=medicine_list,
        med1=med1, med2=med2, med3=med3,
        dose1=dose1, dose2=dose2, dose3=dose3,
        age_range=age_range, symptoms=symptoms,
        acc=MODEL_ACC, cm=MODEL_CM
    )

if __name__ == '__main__':
    # Always retrain at start
    if os.path.exists("overdose_model_multi.pkl"):
        os.remove("overdose_model_multi.pkl")
    train_model()
    app.run(debug=True)

