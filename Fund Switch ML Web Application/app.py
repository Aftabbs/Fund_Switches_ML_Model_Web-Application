import os 
import subprocess
import sys
from flask import Flask, render_template, request, redirect, url_for, send_file, flash
import pandas as pd
import numpy as np
import traceback 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, precision_recall_curve
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)
app.secret_key = 'supersecretkey'

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def calculate_odds_ratios(model, feature_names):
    """Calculate odds ratios for logistic regression coefficients"""
    odds_ratios = pd.Series(np.exp(model.coef_[0]), index=feature_names)
    return odds_ratios

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_data.csv')
        file.save(filename)
        flash('File successfully uploaded')
        return redirect(url_for('train_model'))

@app.route('/train', methods=['GET'])
def train_model():
    try:
        df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_data.csv'))
        print("Data loaded successfully")

        df['AsofDate'] = pd.to_datetime(df['AsofDate'])
        filtered_df = df[df['AsofDate'].dt.year != 2024]

        X = filtered_df.drop(['NewFund', 'PrevAuditor', 'CurrentAuditor', 'AsofDate', 'crd', 'FundID', 'manager_id', 'Auditor Group', 'Switched'], axis=1)
        y = filtered_df['Switched']

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=12)

        lgb = LGBMClassifier(class_weight='balanced', random_state=42)
        lgb.fit(X_train, y_train)
        feature_importances = pd.DataFrame({'feature': X_train.columns, 'importance': lgb.feature_importances_})
        feature_importances = feature_importances.sort_values(by='importance', ascending=False)
        top_features = feature_importances['feature'].head(70).tolist()

        X_train_selected = X_train[top_features]
        X_test_selected = X_test[top_features]

        rf = RandomForestClassifier(class_weight='balanced', random_state=42)
        xgb = XGBClassifier(scale_pos_weight=len(y_train) / sum(y_train), random_state=42)

        ensemble_model = VotingClassifier(estimators=[
            ('lgb', lgb),
            ('rf', rf),
            ('xgb', xgb)
        ], voting='soft')

        ensemble_model.fit(X_train_selected, y_train)
        print("Model trained successfully")

        y_pred_proba = ensemble_model.predict_proba(X_test_selected)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
        y_pred_adjusted = (y_pred_proba >= 0.5859).astype(int)

        accuracy = accuracy_score(y_test, y_pred_adjusted)
        precision = precision_score(y_test, y_pred_adjusted)
        recall = recall_score(y_test, y_pred_adjusted)
        f1 = f1_score(y_test, y_pred_adjusted)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        print(f'Accuracy: {np.round(accuracy * 100,2)}%')
        print(f'Precision: {np.round(precision * 100,2)}%')
        print(f'Recall: {np.round(recall * 100,2)}%')
        print(f'F1-Score: {np.round(f1 * 100,2)}%')
        print(f'ROC-AUC: {np.round(roc_auc * 100,2)}%')

        test_data = filtered_df.loc[X_test.index]
        test_data['Predicted_Switch'] = y_pred_adjusted
        test_data['Probability'] = y_pred_proba

        result_columns = df.columns.tolist() + ['Predicted_Switch', 'Probability']
        test_results = test_data[result_columns]

        data_2024 = df[df['AsofDate'].dt.year == 2024]
        X_2024 = data_2024[top_features]
        y_2024_prob = ensemble_model.predict_proba(X_2024)[:, 1]
        y_2024_pred = (y_2024_prob >= 0.5859).astype(int)
        data_2024['Predicted_Switch'] = y_2024_pred
        data_2024['Probability'] = y_2024_prob
        predictions_2024 = data_2024[result_columns]

        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'Base_FundSwitch_Model_Results.xlsx')
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            test_results.to_excel(writer, sheet_name='Test_Results', index=False)
            predictions_2024.to_excel(writer, sheet_name='2024_Predictions', index=False)
            pd.DataFrame({
                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
                'Score': [accuracy, precision, recall, f1, roc_auc]
            }).to_excel(writer, sheet_name='Metrics', index=False)
            feature_importances.to_excel(writer, sheet_name='Feature_Importance', index=False)

        print("Results saved to 'Base_FundSwitch_Model_Results.xlsx'")

        return render_template('results.html', 
                               accuracy=accuracy, 
                               precision=precision, 
                               recall=recall, 
                               f1=f1, 
                               roc_auc=roc_auc, 
                               download_link=url_for('download_file', filename='Base_FundSwitch_Model_Results.xlsx'))

    except Exception as e:
        print(f"Error: {e}")
        print(traceback.format_exc())
        flash(f"An error occurred: {e}")
        return redirect(url_for('index'))

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), as_attachment=True)

if __name__ == "__main__":
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
