# -*- coding: utf-8 -*-
"""
Salary Prediction Model Training
Chỉ giữ phần training và lưu model, bỏ tất cả visualization
"""

# ============================================================================
# Import Libraries
# ============================================================================
import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV

warnings.filterwarnings('ignore')

print("="*70)
print("SALARY PREDICTION MODEL TRAINING")
print("="*70)

# ============================================================================
# 1. Load và Clean Data
# ============================================================================
print("\n[1/7] Loading data...")
df = pd.read_csv('Salary_Data.csv')

# Chuẩn hóa Education Level
df['Education Level'] = df['Education Level'].replace({
    "Bachelor's Degree": "Bachelor's",
    "Master's Degree": "Master's",
    "phD": "PhD"
})

print(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")

# ============================================================================
# 2. Xử lý Missing Values và Outliers
# ============================================================================
print("\n[2/7] Cleaning data...")

# Loại bỏ missing values
df_clean = df.dropna(subset=['Salary', 'Age', 'Years of Experience']).copy()
print(f"After removing NA: {len(df_clean)} rows")

# Loại bỏ outliers bằng IQR
def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return data[(data[column] < lower) | (data[column] > upper)].index

outliers = set()
for col in ['Salary', 'Age', 'Years of Experience']:
    outliers.update(detect_outliers_iqr(df_clean, col))

df_clean = df_clean.drop(outliers, errors='ignore')
print(f"After removing outliers: {len(df_clean)} rows")

# ============================================================================
# 3. Feature Engineering
# ============================================================================
print("\n[3/7] Feature engineering...")

df_features = df_clean.copy()

# Mã hóa Education Level
education_map = {
    'High School': 1,
    "Bachelor's": 2,
    "Master's": 3,
    'PhD': 4
}
df_features['Education_Encoded'] = df_features['Education Level'].map(education_map)

# One-hot encode Gender (drop_first để tránh multicollinearity)
gender_dummies = pd.get_dummies(df_features['Gender'], prefix='Gender', drop_first=True)
df_features = pd.concat([df_features, gender_dummies], axis=1)

# Tạo polynomial features
df_features['Age_squared'] = df_features['Age'] ** 2
df_features['Exp_squared'] = df_features['Years of Experience'] ** 2

# Tạo interaction feature
df_features['Education_x_Exp'] = (df_features['Education_Encoded'] * 
                                   df_features['Years of Experience'])

# Gộp Job Title (top 10 + Other)
job_counts = df_features['Job Title'].value_counts()
top_n_jobs = job_counts.nlargest(10).index

df_features['Job Title Grouped'] = np.where(
    df_features['Job Title'].isin(top_n_jobs),
    df_features['Job Title'],
    'Other'
)
job_dummies = pd.get_dummies(df_features['Job Title Grouped'], prefix='Job', drop_first=True)
df_features = pd.concat([df_features, job_dummies], axis=1)

# Drop các cột gốc đã encode
df_features = df_features.drop(
    columns=['Job Title', 'Job Title Grouped', 'Gender', 'Education Level'],
    errors='ignore'
)

# Drop remaining NA
df_features = df_features.dropna()

print(f"Final feature set: {df_features.shape[1]} columns")
print(f"Features: {list(df_features.columns)[:10]}...")

# ============================================================================
# 4. Chuẩn bị Data và Scaling
# ============================================================================
print("\n[4/7] Preparing data and scaling...")

# Tách features và target
base_features = [col for col in df_features.columns if col != 'Salary']
X_raw = df_features[base_features].copy()
y = df_features['Salary'].copy()

# Định nghĩa các cột cần scale
numerical_cols_to_scale = [
    'Age', 
    'Years of Experience', 
    'Age_squared', 
    'Exp_squared', 
    'Education_x_Exp'
]

# Tạo và lưu scaler cho mỗi cột
scaler_dict = {}
X_scaled = X_raw.copy()

for col in numerical_cols_to_scale:
    if col in X_raw.columns:
        scaler = StandardScaler()
        scaler.fit(X_raw[[col]])
        scaler_dict[col] = scaler
        X_scaled[col] = scaler.transform(X_raw[[col]])
        print(f"  {col}: mean={scaler.mean_[0]:.2f}, std={scaler.scale_[0]:.2f}")

X = X_scaled
print(f"\nFinal X shape: {X.shape}")
print(f"Scalers created: {len(scaler_dict)}")

# ============================================================================
# 5. Train Multiple Models
# ============================================================================
print("\n[5/7] Training models...")

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    """Train và evaluate model"""
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    print(f"\n  {model_name}:")
    print(f"    Train R²: {train_r2:.4f}")
    print(f"    Test R²:  {test_r2:.4f}")
    print(f"    Test MAE: ${test_mae:,.2f}")
    print(f"    Test RMSE: ${test_rmse:,.2f}")
    
    return {
        'Model': model_name,
        'Train_R2': train_r2,
        'Test_R2': test_r2,
        'Test_MAE': test_mae,
        'Test_RMSE': test_rmse,
        'Overfit_Gap': train_r2 - test_r2,
        'Model_Object': model
    }

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Dictionary để lưu kết quả
all_results = []

# Model 1: Simple Linear Regression
X_train_slr, X_test_slr, y_train_slr, y_test_slr = train_test_split(
    df_features[['Years of Experience']], 
    df_features['Salary'], 
    test_size=0.2, 
    random_state=42
)
model_slr = LinearRegression()
result_slr = evaluate_model(
    model_slr, X_train_slr, y_train_slr, X_test_slr, y_test_slr,
    "Simple Linear Regression"
)
all_results.append(result_slr)

# Model 2: Multiple Linear Regression (3 features)
X_train_mlr = X_train[['Age', 'Years of Experience', 'Education_Encoded']]
X_test_mlr = X_test[['Age', 'Years of Experience', 'Education_Encoded']]

model_mlr = LinearRegression()
result_mlr = evaluate_model(
    model_mlr, X_train_mlr, y_train, X_test_mlr, y_test,
    "Multiple Linear Regression"
)
all_results.append(result_mlr)

# Model 3: Polynomial Regression (Degree 2)
print("\n" + "="*70)
print("Polynomial Regression (Degree 2)")
print("="*70)

# Chuẩn bị X ban đầu cho Polynomial
X_poly2_base = df_clean.drop(columns=['Salary', 'Job Title', 'Gender', 'Education Level'], errors='ignore')
X_poly2_base['Education_Encoded'] = df_clean['Education Level'].map(education_map)

# Thêm dummy variables cho Gender
gender_dummies_poly = pd.get_dummies(df_clean['Gender'], prefix='Gender', drop_first=True)
X_poly2_base = pd.concat([X_poly2_base, gender_dummies_poly], axis=1)

# Gộp Job Title và tạo dummy variables
X_poly2_base['Job Title Grouped'] = np.where(
    df_clean['Job Title'].isin(top_n_jobs),
    df_clean['Job Title'],
    'Other'
)
job_dummies_poly = pd.get_dummies(X_poly2_base['Job Title Grouped'], prefix='Job', drop_first=True)
X_poly2_base = pd.concat([X_poly2_base, job_dummies_poly], axis=1)
X_poly2_base = X_poly2_base.drop(columns=['Job Title Grouped'], errors='ignore').dropna()

# Tạo features đa thức bậc 2 cho Age và Years of Experience
poly2_transformer = PolynomialFeatures(degree=2, include_bias=False)
poly_num_features = poly2_transformer.fit_transform(X_poly2_base[['Age', 'Years of Experience']])
poly_num_feature_names = poly2_transformer.get_feature_names_out(['Age', 'Years of Experience'])

# DataFrame cho các features đa thức
X_poly2_num = pd.DataFrame(poly_num_features, columns=poly_num_feature_names, index=X_poly2_base.index)

# Kết hợp: loại bỏ các cột gốc và thêm các cột đa thức
X_poly2 = X_poly2_base.drop(columns=['Age', 'Years of Experience']).copy()
X_poly2 = pd.concat([X_poly2, X_poly2_num], axis=1)

# Scaling các cột số - FIXED: Tạo scaler riêng cho từng cột
numerical_cols_for_poly_scaled = [col for col in X_poly2.columns
                                  if X_poly2[col].dtype in ['int64', 'float64']
                                  and not col.startswith('Gender_')
                                  and not col.startswith('Job_')
                                  and col != 'Education_Encoded']

# Tạo dictionary để lưu scaler cho từng cột
scaler_poly2_dict = {}
X_poly2_scaled = X_poly2.copy()
for col in numerical_cols_for_poly_scaled:
    scaler = StandardScaler()
    X_poly2_scaled[col] = scaler.fit_transform(X_poly2[[col]])
    scaler_poly2_dict[col] = scaler
    print(f"  Poly {col}: mean={scaler.mean_[0]:.2f}, std={scaler.scale_[0]:.2f}")

# Đảm bảo y khớp index với X
y_poly2 = df_clean.loc[X_poly2_scaled.index, 'Salary']

# Chia dữ liệu train/test
X_train_poly2, X_test_poly2, y_train_poly2, y_test_poly2 = train_test_split(
    X_poly2_scaled, y_poly2, test_size=0.2, random_state=42
)

# Huấn luyện mô hình
model_poly2 = LinearRegression()
result_poly2 = evaluate_model(model_poly2, X_train_poly2, y_train_poly2,
                              X_test_poly2, y_test_poly2, "Polynomial Regression (Degree 2)")

# Cross-Validation 5-fold
cv_scores_poly2 = cross_val_score(model_poly2, X_poly2_scaled, y_poly2, cv=5, scoring='r2', n_jobs=-1)
print(f"\n  R² trung bình (5-Fold CV): {cv_scores_poly2.mean():.4f} (+/- {cv_scores_poly2.std():.4f})")
result_poly2['CV_R2_Mean'] = cv_scores_poly2.mean()
result_poly2['CV_R2_Std'] = cv_scores_poly2.std()
all_results.append(result_poly2)

# Model 4: Ridge Regression
alphas = np.logspace(-3, 3, 100)
ridge_cv = RidgeCV(alphas=alphas, cv=5, scoring='neg_mean_squared_error')
ridge_cv.fit(X_train, y_train)
print(f"\n  Ridge optimal alpha: {ridge_cv.alpha_:.6f}")
result_ridge = evaluate_model(
    ridge_cv, X_train, y_train, X_test, y_test,
    "Ridge Regression"
)
all_results.append(result_ridge)

# Model 5: Lasso Regression
lasso_cv = LassoCV(
    alphas=np.logspace(-4, 1, 100),
    cv=5,
    random_state=42,
    max_iter=10000,
    n_jobs=-1
)
lasso_cv.fit(X_train, y_train)
print(f"\n  Lasso optimal alpha: {lasso_cv.alpha_:.6f}")
model_lasso = Lasso(alpha=lasso_cv.alpha_, max_iter=10000, random_state=42)
result_lasso = evaluate_model(
    model_lasso, X_train, y_train, X_test, y_test,
    "Lasso Regression"
)
print(f"  Non-zero coefficients: {np.sum(model_lasso.coef_ != 0)}")
all_results.append(result_lasso)

# ============================================================================
# 6. Select Best Model
# ============================================================================
print("\n[6/7] Selecting best model...")

results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values(by='Test_R2', ascending=False).reset_index(drop=True)

print("\n" + "="*70)
print("MODEL COMPARISON")
print("="*70)
print(results_df[['Model', 'Test_R2', 'Test_MAE', 'Test_RMSE', 'Overfit_Gap']].to_string(index=False))

best_model_name = results_df.iloc[0]['Model']
best_model_object = results_df.iloc[0]['Model_Object']

print("\n" + "="*70)
print(f"BEST MODEL: {best_model_name}")
print(f"  Test R²: {results_df.iloc[0]['Test_R2']:.4f}")
print(f"  Test MAE: ${results_df.iloc[0]['Test_MAE']:,.2f}")
print(f"  Test RMSE: ${results_df.iloc[0]['Test_RMSE']:,.2f}")
print("="*70)

# ============================================================================
# 7. Save Model and Preprocessing Objects
# ============================================================================
print("\n[7/7] Saving model and preprocessing objects...")

# Determine feature columns based on best model
if best_model_name == "Simple Linear Regression":
    feature_columns = ['Years of Experience']
elif best_model_name == "Polynomial Regression (Degree 2)":
    feature_columns = list(X_poly2_scaled.columns)
    # Lưu thêm poly transformer và scaler dict cho poly model
    joblib.dump(poly2_transformer, 'poly2_transformer.pkl')
    joblib.dump(scaler_poly2_dict, 'scaler_poly2_dict.pkl')
    print(f"  Polynomial features: {len(feature_columns)} columns")
    print(f"  Polynomial scalers: {len(scaler_poly2_dict)} scalers")
else:
    feature_columns = list(X.columns)

# Save all objects
joblib.dump(best_model_object, 'best_salary_model.pkl')
joblib.dump(education_map, 'education_map.pkl')
joblib.dump(top_n_jobs, 'top_n_jobs.pkl')
joblib.dump(feature_columns, 'model_features_order.pkl')
joblib.dump(scaler_dict, 'scaler_dict.pkl')
joblib.dump(best_model_name, 'best_model_name.pkl')

print("\n" + "="*70)
print("SAVED FILES")
print("="*70)
print("best_salary_model.pkl")
print("best_model_name.pkl")
print(f"education_map.pkl ({len(education_map)} levels: {list(education_map.keys())})")
print(f"top_n_jobs.pkl ({len(top_n_jobs)} jobs)")
print(f"model_features_order.pkl ({len(feature_columns)} features)")
print(f"scaler_dict.pkl ({len(scaler_dict)} scalers)")

if best_model_name == "Polynomial Regression (Degree 2)":
    print("poly2_transformer.pkl")
    print(f"scaler_poly2_dict.pkl ({len(scaler_poly2_dict)} scalers)")
    print("\nPolynomial Scalers:")
    for col, scaler in scaler_poly2_dict.items():
        print(f"  {col}: mean={scaler.mean_[0]:.2f}, std={scaler.scale_[0]:.2f}")

print("\n" + "="*70)
print("SCALER VERIFICATION (IMPORTANT)")
print("="*70)
for col, scaler in scaler_dict.items():
    print(f"  {col}:")
    print(f"    mean = {scaler.mean_[0]:.2f}")
    print(f"    std  = {scaler.scale_[0]:.2f}")

print("\n" + "="*70)
print("TRAINING COMPLETE")
print("="*70)