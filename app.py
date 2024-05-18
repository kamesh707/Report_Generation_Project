from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from ydata_profiling import ProfileReport

app = Flask(__name__)

# Dummy passwords for users
passwords = {
    'user1': 'password1',
    'user2': 'password2',
    'user3': 'password3',
    'user4': 'password4',
    'user5': 'password5',
    'user6': 'password6',
    'user7': 'password7',
    'user8': 'password8',
    'user9': 'password9',
    'user10': 'password10'
}

# Load data
data = pd.read_csv("https://raw.githubusercontent.com/sunnysavita10/credit_card_pw_hindi/main/creditCardFraud_28011964_120214.csv")

# Prepare data for modeling
X = data.drop(labels=['default payment next month'], axis=1)
y = data['default payment next month']

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train models
rf_model = RandomForestClassifier()
rf_model.fit(X_scaled, y)

xgb_model = XGBClassifier()
xgb_model.fit(X_scaled, y)

# Split data into ten equal parts for ten users
split_data = [data.iloc[i:i+100] for i in range(0, len(data), 100)]

@app.route('/')
def index():
    # Redirect to login page if not logged in
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        # Check if username and password are correct
        if username in passwords and passwords[username] == password:
            session['username'] = username
            return redirect(url_for('index'))
        else:
            return render_template('login.html', message='Invalid username or password.')
    return render_template('login.html')

@app.route('/logout')
def logout():
    # Clear the session
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/predict', methods=['POST'])
def predict():
    # Check if user is logged in
    if 'username' not in session:
        return redirect(url_for('login'))

    # Extract input values from the form
    input_values = [float(request.form[feature]) for feature in X.columns]
    scaled_input = scaler.transform([input_values])

    # Predictions from Random Forest and XGBoost models
    prediction_rf = rf_model.predict(scaled_input)[0]
    prediction_xgb = xgb_model.predict(scaled_input)[0]

    # Convert predictions to human-readable format
    prediction_rf_human = "WILL PAY" if prediction_rf == 1 else "WON'T PAY"
    prediction_xgb_human = "WILL PAY" if prediction_xgb == 1 else "WON'T PAY"
    
    # Overall prediction combining both models
    overall_prediction = "WILL PAY" if prediction_rf + prediction_xgb >= 1 else "WON'T PAY"

    # Get the user index based on which subset of data to use
    user_index = int(request.form['user_index'])
    user_data = split_data[user_index]

    # Generate profile report for user data
    profile = ProfileReport(user_data, title="User Data Profile Report")
    profile_html = profile.to_html()

    return render_template('result.html', prediction_rf=prediction_rf_human, prediction_xgb=prediction_xgb_human, overall_prediction=overall_prediction, profile_html=profile_html)
if __name__ == '__main__':
    app.secret_key = 'super_secret_key'
    app.run(debug=True)