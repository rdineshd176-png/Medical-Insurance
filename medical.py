# =============================
# Medical Insurance Prediction Project
# =============================

# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

# ----------------------------
# Step 1: Load Dataset
# ----------------------------
# Use raw string for file path
file_path = r"D:\100 days of 100 projects\medical insurance\medical_insurance_with_profession_salary_lifestyle.xlsx"

# Check if file exists
if not os.path.exists(file_path):
    print(f"❌ Error: File not found at {file_path}")
    print("Please check the file path and try again.")
    exit()

try:
    data = pd.read_excel(file_path)
    print("✅ Dataset loaded successfully!")
    print(f"Dataset shape: {data.shape}")
except Exception as e:
    print(f"❌ Error loading file: {e}")
    exit()

# Drop unnecessary columns safely
columns_to_drop = ['name', 'charges_original', 'region']
for col in columns_to_drop:
    if col in data.columns:
        data = data.drop(columns=[col])
        print(f"✅ Dropped column: {col}")

print(f"Columns after dropping: {list(data.columns)}")

# Standardize categorical values to lowercase
categorical_columns = ['sex', 'smoker', 'past_conditions', 'profession']
for col in categorical_columns:
    if col in data.columns:
        data[col] = data[col].astype(str).str.lower()
        print(f"✅ Standardized {col} to lowercase")

# Encode categorical variables
le_dict = {}
for col in categorical_columns:
    if col in data.columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        le_dict[col] = le   # save encoder for later use
        print(f"✅ Encoded column: {col}")
        print(f"   Classes: {list(le.classes_)}")
    else:
        print(f"⚠️  Column {col} not found in dataset")

# Split features and target
if 'charges' not in data.columns:
    print("❌ Error: 'charges' column not found in dataset")
    exit()

X = data.drop('charges', axis=1)
y = data['charges']

print(f"Features: {list(X.columns)}")
print(f"Target: charges")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set: {X_train.shape}")
print(f"Testing set: {X_test.shape}")

# Train model
print("🔄 Training Random Forest model...")
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluate model
y_pred = rf.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"✅ Model trained successfully!")
print(f"📊 Model Performance:")
print(f"   Mean Absolute Error: ₹{mae:.2f}")
print(f"   R² Score: {r2:.4f}")

# Save model and encoders
try:
    joblib.dump(rf, 'insurance_model.pkl')
    joblib.dump(le_dict, 'encoders.pkl')
    print("✅ Model and encoders saved successfully!")
except Exception as e:
    print(f"❌ Error saving model: {e}")

# ----------------------------
# Helper Functions
# ----------------------------
def encode_past_conditions(condition_text, le):
    """Encode past conditions text input"""
    if not condition_text:
        return le.transform(['none'])[0]
    
    condition_text = condition_text.lower().strip()
    
    # Common conditions mapping
    conditions_map = {
        'none': 'none',
        'no': 'none',
        'healthy': 'none',
        'asthma': 'asthma',
        'diabetes': 'diabetes',
        'hypertension': 'hypertension',
        'high blood pressure': 'hypertension',
        'heart disease': 'heart_disease',
        'heart problem': 'heart_disease',
        'blood pressure': 'hypertension'
    }
    
    # Check if input matches any known condition
    for key, value in conditions_map.items():
        if key in condition_text:
            condition_text = value
            break
    
    # If condition not in trained classes, default to 'none'
    if condition_text not in le.classes_:
        condition_text = 'none'
    
    return le.transform([condition_text])[0]

def safe_encode(value, encoder, default_index=0):
    """Safely encode a value with fallback to default"""
    try:
        value = str(value).lower()
        if value in encoder.classes_:
            return encoder.transform([value])[0]
        else:
            print(f"⚠️  Value '{value}' not in {encoder.classes_}, using default index {default_index}")
            return default_index
    except Exception as e:
        print(f"⚠️  Encoding error for '{value}': {e}, using default index {default_index}")
        return default_index

def give_suggestions(age, bmi, smoker, steps, exercise, sleep, marital_status):
    """Generate health suggestions"""
    suggestions = []
    if smoker == 1:  
        suggestions.append("Quit smoking to reduce insurance premium.")
    if bmi > 30:
        suggestions.append("Maintain a healthy diet and exercise to reduce BMI.")
    if age > 50:
        suggestions.append("Go for regular medical check-ups to avoid high risk charges.")
    if steps < 5000:
        suggestions.append("Increase your daily steps to improve overall health.")
    if exercise < 2:
        suggestions.append("Do at least 2-3 hours of exercise per week.")
    if sleep < 6:
        suggestions.append("Increase your sleep hours for better recovery.")
    if marital_status.lower() == "single":
        suggestions.append("Consider family insurance plans if you marry in future.")

    suggestions.append("Maintaining good and healthy habits will reduce your insurance premium over time.")
    return suggestions

# ----------------------------
# Prediction Function
# ----------------------------
def predict_premium(user_data, name, marital_status):
    # Predict base premium
    premium = rf.predict(user_data)[0]

    # Extract habits (updated indices)
    age = user_data[0,0]
    bmi = user_data[0,2]
    smoker = user_data[0,4]
    steps = user_data[0,8]    # updated index
    exercise = user_data[0,9] # updated index
    sleep = user_data[0,10]   # updated index

    # Apply healthy lifestyle discount
    discount = 0
    if smoker == 0: discount += 0.05
    if bmi < 25: discount += 0.05
    if steps >= 7000: discount += 0.05
    if exercise >= 1.5: discount += 0.05
    if sleep >= 7: discount += 0.05

    # Apply discount (max 25%)
    premium = premium * (1 - min(discount, 0.25))

    # Split insurance
    normal_insurance = premium * 0.6
    emergency_insurance = premium * 0.4

    # Suggestions
    suggestions = give_suggestions(age, bmi, smoker, steps, exercise, sleep, marital_status)

    # Output
    print("\n===================================")
    print(f"👤 Insurance Report for {name}")
    print("===================================")
    print("💰 Predicted Insurance Premium: ₹{:.2f}".format(premium))
    print("➡ Normal Insurance (60%): ₹{:.2f}".format(normal_insurance))
    print("➡ Emergency Insurance (40%): ₹{:.2f}".format(emergency_insurance))
    print(f"🎁 Lifestyle Discount Applied: {min(discount * 100, 25):.1f}%")
    print("\n✅ Suggestions for You:")
    for s in suggestions:
        print("- " + s)
    print("===================================\n")

# ----------------------------
# Main Loop for Multiple People
# ----------------------------
print("\n" + "="*50)
print("=== Medical Insurance Prediction System ===")
print("="*50)

while True:
    try:
        print("\n--- Enter Details ---")
        name = input("Enter your name: ")
        age = int(input("Enter your age: "))
        sex = input("Enter your sex (male/female): ").lower()
        marital_status = input("Enter marital status (single/married): ").lower()
        salary = int(input("Enter your annual salary: "))
        profession = input("Enter your profession: ").lower()
        bmi = float(input("Enter your BMI: "))
        children = int(input("Enter number of children (or 0 if none): "))
        smoker = input("Are you a smoker? (yes/no): ").lower()
        past_conditions = input("Enter past medical condition (type any condition or 'none'): ")
        daily_steps = int(input("Enter your average daily steps: "))
        exercise_hours = float(input("Enter exercise hours per week: "))
        sleep_hours = float(input("Enter sleep hours per day: "))

        # Encode categorical inputs using safe encoding
        sex_enc = safe_encode(sex, le_dict['sex'])
        smoker_enc = safe_encode(smoker, le_dict['smoker'])
        past_cond_enc = encode_past_conditions(past_conditions, le_dict['past_conditions'])
        profession_enc = safe_encode(profession, le_dict['profession'])

        # Build input array
        input_data = np.array([[age, sex_enc, bmi, children, smoker_enc, past_cond_enc,
                                profession_enc, salary, daily_steps, exercise_hours, sleep_hours]])

        # Predict and show report
        predict_premium(input_data, name, marital_status)

        # Ask for another person
        cont = input("Do you want to enter another person? (yes/no): ").lower()
        if cont != "yes":
            print("\n✅ Thank you for using the Medical Insurance Prediction System!")
            break
            
    except ValueError:
        print("❌ Invalid input! Please enter correct values.")
    except Exception as e:
        print(f"❌ An error occurred: {e}")