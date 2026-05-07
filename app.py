from flask import Flask, render_template, request
import pickle
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load model & preprocessor
model = pickle.load(open('rf_multioutput_model.pkl', 'rb'))
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))

# Load label encoders
le_bug_category = pickle.load(open('le_bug_category.pkl', 'rb'))
le_severity = pickle.load(open('le_severity.pkl', 'rb'))
le_suggested_fix = pickle.load(open('le_suggested_fix.pkl', 'rb'))
le_bug_domain = pickle.load(open('le_bug_domain.pkl', 'rb'))
le_tech_stack = pickle.load(open('le_tech_stack.pkl', 'rb'))
le_environment = pickle.load(open('le_environment.pkl', 'rb'))
le_developer_role = pickle.load(open('le_developer_role.pkl', 'rb'))
le_root_cause = pickle.load(open('le_root_cause.pkl', 'rb'))

def encoder_options(le):
    return list(le.classes_)


@app.route('/')
def index():
    return render_template(
        'index.html',
        bug_domains=encoder_options(le_bug_domain),
        tech_stacks=encoder_options(le_tech_stack),
        environments=encoder_options(le_environment),
        developer_roles=encoder_options(le_developer_role),
        root_causes=encoder_options(le_root_cause)
    )


@app.route('/predict', methods=['POST'])
def predict():
    title = request.form['title']
    description = request.form['description']
    explanation = request.form['explanation']

    error_code = int(request.form['error_code'])
    bug_domain_raw = request.form['bug_domain']
    tech_stack_raw = request.form['tech_stack']
    environment_raw = request.form['environment']
    developer_role_raw = request.form['developer_role']
    root_cause_raw = request.form['root_cause']

    bug_domain = le_bug_domain.transform([bug_domain_raw])[0]
    tech_stack = le_tech_stack.transform([tech_stack_raw])[0]
    environment = le_environment.transform([environment_raw])[0]
    developer_role = le_developer_role.transform([developer_role_raw])[0]
    root_cause = le_root_cause.transform([root_cause_raw])[0]

    today = datetime.now()
    year, month, day = today.year, today.month, today.day

    input_df = pd.DataFrame([{
        'title': title,
        'description': description,
        'explanation': explanation,
        'error_code': error_code,
        'bug_domain': bug_domain,
        'tech_stack': tech_stack,
        'environment': environment,
        'developer_role': developer_role,
        'root_cause': root_cause,
        'year': year,
        'month': month,
        'day': day
    }])

    transformed = preprocessor.transform(input_df)
    prediction = model.predict(transformed)[0]

    bug_category = le_bug_category.inverse_transform([prediction[0]])[0]
    severity = le_severity.inverse_transform([prediction[1]])[0]
    suggested_fix = le_suggested_fix.inverse_transform([prediction[2]])[0]

    report = {
        'title': title,
        'description': description,
        'explanation': explanation,
        'error_code': error_code,
        'bug_domain': bug_domain_raw,
        'tech_stack': tech_stack_raw,
        'environment': environment_raw,
        'developer_role': developer_role_raw,
        'root_cause': root_cause_raw,
        'bug_category': bug_category,
        'severity': severity,
        'suggested_fix': suggested_fix,
        'submitted_at': today.isoformat()
    }

    return render_template(
        'result.html',
        bug_category=bug_category,
        severity=severity,
        suggested_fix=suggested_fix,
        report=report
    )


@app.route('/history')
def history():
    return render_template('history.html')


if __name__ == '__main__':
    app.run(debug=True)
