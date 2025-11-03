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
# Correct file path - use one of these options:

# Option 1: Use raw string (recommended)
file_path = r"D:\100 days of 100 projects\medical insurance\medical_insurance_with_profession_salary_lifestyle.xlsx"

# Option 2: Use forward slashes
# file_path = "D:/100 days of 100 projects/medical insurance/medical_insurance_with_profession_salary_lifestyle.xlsx"

# Option 3: Use double backslashes
# file_path = "D:\\100 days of 100 projects\\medical insurance\\medical_insurance_with_profession_salary_lifestyle.xlsx"

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

# Encode categorical variables (removed region)
le_dict = {}
categorical_columns = ['sex', 'smoker', 'past_conditions', 'profession']

for col in categorical_columns:
    if col in data.columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        le_dict[col] = le   # save encoder for later use
        print(f"✅ Encoded column: {col}")
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
# Suggestion Function
# ----------------------------
def give_suggestions(age, bmi, smoker, steps, exercise, sleep, marital_status):
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

    # ✅ Always encourage
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
    if smoker == 0: discount += 0.05   # 5% discount if non-smoker
    if bmi < 25: discount += 0.05      # 5% discount for healthy BMI
    if steps >= 7000: discount += 0.05 # 5% discount for active lifestyle
    if exercise >= 1.5: discount += 0.05 # 5% discount for regular exercise
    if sleep >= 7: discount += 0.05    # 5% discount for good sleep

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
        sex = input("Enter your sex (male/female): ")
        marital_status = input("Enter marital status (single/married): ")
        salary = int(input("Enter your annual salary: "))
        profession = input("Enter your profession: ")
        bmi = float(input("Enter your BMI: "))
        children = int(input("Enter number of children (or 0 if none): "))
        smoker = input("Are you a smoker? (yes/no): ")
        past_conditions = input("Enter past medical condition (none/asthma/hypertension/etc.): ")
        daily_steps = int(input("Enter your average daily steps: "))
        exercise_hours = float(input("Enter exercise hours per week: "))
        sleep_hours = float(input("Enter sleep hours per day: "))

        # Encode categorical inputs using saved encoders
        sex_enc = le_dict['sex'].transform([sex])[0] if sex in le_dict['sex'].classes_ else 0
        smoker_enc = le_dict['smoker'].transform([smoker])[0] if smoker in le_dict['smoker'].classes_ else 0
        past_cond_enc = le_dict['past_conditions'].transform([past_conditions])[0] if past_conditions in le_dict['past_conditions'].classes_ else 0
        profession_enc = le_dict['profession'].transform([profession])[0] if profession in le_dict['profession'].classes_ else 0

        # Build input array (removed region)
        input_data = np.array([[age, sex_enc, bmi, children, smoker_enc, past_cond_enc,
                                profession_enc, salary, daily_steps, exercise_hours, sleep_hours]])

        # Predict and show report
        predict_premium(input_data, name, marital_status)

        # Ask for another person
        cont = input("Do you want to enter another person? (yes/no): ")
        if cont.lower() != "yes":
            print("\n✅ Thank you for using the Medical Insurance Prediction System!")
            break
            
    except ValueError:
        print("❌ Invalid input! Please enter correct values.")
    except Exception as e:
        print(f"❌ An error occurred: {e}")