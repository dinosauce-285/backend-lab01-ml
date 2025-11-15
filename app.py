from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import warnings
import traceback

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

model = None
scaler_dict = None
education_map = None
top_n_jobs = None
FEATURE_COLUMNS_ORDER = None

def load_models():
    """Load all saved model objects"""
    global model, scaler_dict, education_map, top_n_jobs, FEATURE_COLUMNS_ORDER
    
    try:
        model = joblib.load('best_salary_model.pkl')
        scaler_dict = joblib.load('scaler_dict.pkl')
        education_map = joblib.load('education_map.pkl')
        top_n_jobs = joblib.load('top_n_jobs.pkl')
        FEATURE_COLUMNS_ORDER = joblib.load('model_features_order.pkl')
        
        print("="*70)
        print("MODEL LOADING STATUS")
        print("="*70)
        print(f"Model: {type(model).__name__}")
        print(f"Scalers: {list(scaler_dict.keys())}")
        
        print("\nScaler parameters:")
        for col, scaler in scaler_dict.items():
            print(f"  {col}: mean={scaler.mean_[0]:.2f}, std={scaler.scale_[0]:.2f}")
        
        print(f"\nEducation levels: {list(education_map.keys())}")
        print(f"Top jobs ({len(top_n_jobs)}): {list(top_n_jobs)}")
        print(f"Features ({len(FEATURE_COLUMNS_ORDER)})")
        print("="*70)
        print("All models loaded successfully!\n")
        
    except Exception as e:
        print("="*70)
        print("ERROR LOADING MODELS")
        print("="*70)
        print(f"Error: {str(e)}")
        traceback.print_exc()
        print("="*70)

load_models()

@app.after_request
def after_request(response):
    """Add CORS headers"""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

@app.route('/', methods=['GET'])
def home():
    """Health check endpoint"""
    return jsonify({
        'status': 'running',
        'service': 'Salary Prediction API',
        'model_loaded': model is not None,
        'features_count': len(FEATURE_COLUMNS_ORDER) if FEATURE_COLUMNS_ORDER else 0
    })

@app.route('/predict_salary', methods=['OPTIONS'])
def handle_options():
    """Handle CORS preflight"""
    return '', 204

@app.route('/predict_salary', methods=['POST'])
def predict_salary():
    """Main prediction endpoint"""
    print("\n" + "="*70)
    print("NEW PREDICTION REQUEST")
    print("="*70)
    
    if any(x is None for x in [model, scaler_dict, education_map, top_n_jobs, FEATURE_COLUMNS_ORDER]):
        print("ERROR: Models not loaded properly")
        return jsonify({'error': 'Models not loaded properly'}), 500

    try:
        data = request.get_json(force=True)
        print(f"Request data: {data}")

        required_fields = ['Age', 'Years of Experience', 'Education Level', 'Job Title', 'Gender']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        input_df = pd.DataFrame(columns=FEATURE_COLUMNS_ORDER)
        input_df.loc[0] = 0.0 

        age = float(data['Age'])
        years_exp = float(data['Years of Experience'])
        education = data['Education Level']
        job = data['Job Title']
        gender = data['Gender']

        print(f"\nInput values:")
        print(f"  Age: {age}")
        print(f"  Experience: {years_exp} years")
        print(f"  Education: {education}")
        print(f"  Job: {job}")
        print(f"  Gender: {gender}")

        edu_encoded = education_map.get(education)
        if edu_encoded is None:
            return jsonify({
                'error': f'Invalid education level: {education}',
                'valid_levels': list(education_map.keys())
            }), 400
        
        if 'Education_Encoded' in input_df.columns:
            input_df.loc[0, 'Education_Encoded'] = float(edu_encoded)

        if 'Age' in input_df.columns:
            input_df.loc[0, 'Age'] = age
        if 'Years of Experience' in input_df.columns:
            input_df.loc[0, 'Years of Experience'] = years_exp
        if 'Age^2' in input_df.columns:
            input_df.loc[0, 'Age^2'] = age ** 2
        if 'Years of Experience^2' in input_df.columns:
            input_df.loc[0, 'Years of Experience^2'] = years_exp ** 2
        if 'Age Years of Experience' in input_df.columns:
            input_df.loc[0, 'Age Years of Experience'] = age * years_exp

        if 'Age_squared' in input_df.columns:
            input_df.loc[0, 'Age_squared'] = age ** 2
        if 'Exp_squared' in input_df.columns:
            input_df.loc[0, 'Exp_squared'] = years_exp ** 2
        if 'Education_x_Exp' in input_df.columns:
            input_df.loc[0, 'Education_x_Exp'] = edu_encoded * years_exp

        print("\nScaling features:")
        for col in scaler_dict.keys():
            if col in input_df.columns:
                original_val = input_df.loc[0, col]
                scaled_val = scaler_dict[col].transform([[original_val]])[0][0]
                input_df.loc[0, col] = scaled_val
                print(f"  {col}: {original_val:.2f} -> {scaled_val:.4f}")

        if 'Gender_Male' in input_df.columns:
            input_df.loc[0, 'Gender_Male'] = 1.0 if gender == 'Male' else 0.0
        if 'Gender_Other' in input_df.columns:
            input_df.loc[0, 'Gender_Other'] = 1.0 if gender == 'Other' else 0.0

        job_grouped = job if job in top_n_jobs else 'Other'
        print(f"\nJob encoding: '{job}' -> '{job_grouped}'")
        
        for col in input_df.columns:
            if col.startswith('Job_'):
                job_name = col[4:]  
                input_df.loc[0, col] = 1.0 if job_name == job_grouped else 0.0

        print(f"\nDataFrame ready:")
        print(f"  Shape: {input_df.shape}")
        print(f"  Columns match: {list(input_df.columns) == FEATURE_COLUMNS_ORDER}")
        
        prediction = model.predict(input_df)
        result = round(float(prediction[0]), 2)
        
        print(f"\nPrediction: ${result:,.2f}")
        print("="*70 + "\n")
        
        return jsonify({'predicted_salary': result})

    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        traceback.print_exc()
        print("="*70 + "\n")
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

if __name__ == '__main__':
    print("\n" + "="*70)
    print("FLASK SERVER STARTING")
    print("="*70)
    print("Server: http://127.0.0.1:5000")
    print("Endpoints:")
    print("  GET  /              - Health check")
    print("  POST /predict_salary - Predict salary")
    print("="*70 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)