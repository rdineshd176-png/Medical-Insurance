from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import os

app = Flask(__name__)

class InsurancePredictor:
    def __init__(self):
        self.model = None
        self.le_dict = {}
        self.load_model()
    
    def load_model(self):
        """Load the trained model and encoders from separate .pkl files"""
        try:
            # Load the model
            if os.path.exists('insurance_model.pkl'):
                self.model = joblib.load('insurance_model.pkl')
                print("✅ Model loaded successfully!")
            else:
                print("❌ insurance_model.pkl not found!")
                return False
            
            # Load the encoders
            if os.path.exists('encoders.pkl'):
                self.le_dict = joblib.load('encoders.pkl')
                print("✅ Encoders loaded successfully!")
                # Print available classes for debugging
                for col, le in self.le_dict.items():
                    print(f"   {col}: {list(le.classes_)}")
            else:
                print("❌ encoders.pkl not found!")
                return False
                
            return True
            
        except Exception as e:
            print(f"❌ Error loading model/encoders: {str(e)}")
            return False
    
    def safe_encode(self, value, encoder, default_index=0):
        """Safely encode a value with fallback to default"""
        try:
            if value in encoder.classes_:
                return encoder.transform([value])[0]
            else:
                print(f"⚠️  Value '{value}' not in {encoder.classes_}, using default index {default_index}")
                return default_index
        except Exception as e:
            print(f"⚠️  Encoding error for '{value}': {e}, using default index {default_index}")
            return default_index
    
    def encode_past_conditions(self, condition_text):
        """Encode past conditions text input"""
        le = self.le_dict['past_conditions']
        
        if not condition_text:
            return self.safe_encode('none', le)
        
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
        
        # Use safe_encode to handle unknown conditions
        return self.safe_encode(condition_text, le)
    
    def give_suggestions(self, age, bmi, smoker, steps, exercise, sleep, marital_status):
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
    
    def predict(self, input_data, name, marital_status):
        """Make prediction and generate report"""
        try:
            # Predict base premium
            premium = self.model.predict(input_data)[0]

            # Extract features for discount calculation (updated indices)
            age = input_data[0, 0]
            bmi = input_data[0, 2]
            smoker = input_data[0, 4]
            steps = input_data[0, 8]    # updated index
            exercise = input_data[0, 9] # updated index
            sleep = input_data[0, 10]   # updated index

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

            # Generate suggestions
            suggestions = self.give_suggestions(age, bmi, smoker, steps, exercise, sleep, marital_status)

            return {
                'premium': premium,
                'normal_insurance': normal_insurance,
                'emergency_insurance': emergency_insurance,
                'suggestions': suggestions,
                'discount_applied': f"{min(discount * 100, 25):.1f}%"
            }
            
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return None

# Initialize predictor
predictor = InsurancePredictor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        name = request.form['name']
        age = int(request.form['age'])
        sex = request.form['sex'].lower()  # Convert to lowercase for consistency
        marital_status = request.form['marital_status'].lower()
        salary = int(request.form['salary'])
        profession = request.form['profession']
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        smoker = request.form['smoker'].lower()
        past_conditions = request.form['past_conditions']
        daily_steps = int(request.form['daily_steps'])
        exercise_hours = float(request.form['exercise_hours'])
        sleep_hours = float(request.form['sleep_hours'])

        # Encode categorical inputs with safe encoding
        try:
            sex_enc = predictor.safe_encode(sex, predictor.le_dict['sex'])
            smoker_enc = predictor.safe_encode(smoker, predictor.le_dict['smoker'])
            past_cond_enc = predictor.encode_past_conditions(past_conditions)
            profession_enc = predictor.safe_encode(profession, predictor.le_dict['profession'])
            
            print(f"🔍 Encoding results:")
            print(f"   Sex: '{sex}' -> {sex_enc}")
            print(f"   Smoker: '{smoker}' -> {smoker_enc}")
            print(f"   Past Conditions: '{past_conditions}' -> {past_cond_enc}")
            print(f"   Profession: '{profession}' -> {profession_enc}")
            
        except Exception as e:
            error_msg = f"Error encoding input: {str(e)}. Available options - "
            error_msg += f"Sex: {list(predictor.le_dict['sex'].classes_)}, "
            error_msg += f"Smoker: {list(predictor.le_dict['smoker'].classes_)}, "
            error_msg += f"Profession: {list(predictor.le_dict['profession'].classes_)}"
            return render_template('index.html', error=error_msg)

        # Create input array
        input_data = np.array([[age, sex_enc, bmi, children, smoker_enc, past_cond_enc,
                              profession_enc, salary, daily_steps, exercise_hours, sleep_hours]])

        # Make prediction
        result = predictor.predict(input_data, name, marital_status)
        
        if result:
            return render_template('result.html', 
                                 name=name,
                                 premium=result['premium'],
                                 normal_insurance=result['normal_insurance'],
                                 emergency_insurance=result['emergency_insurance'],
                                 suggestions=result['suggestions'],
                                 discount=result['discount_applied'])
        else:
            return render_template('index.html', error="Prediction failed. Please try again.")
            
    except Exception as e:
        return render_template('index.html', error=f"Error processing request: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)