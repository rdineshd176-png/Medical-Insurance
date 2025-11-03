from flask import Flask, request, jsonify
import joblib
import os
import sys

app = Flask(__name__)

class LightweightPredictor:
    def __init__(self):
        self.model = None
        self.le_dict = {}
        self.load_model()
    
    def load_model(self):
        """Load model without numpy/pandas"""
        try:
            self.model = joblib.load('insurance_model.pkl')
            self.le_dict = joblib.load('encoders.pkl')
            print("✅ Model loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def manual_predict(self, input_features):
        """Manual prediction without scikit-learn (if model loading fails)"""
        # Simple rule-based fallback
        base_premium = 20000
        age_factor = input_features[0] * 100  # age
        bmi_factor = input_features[2] * 200  # bmi
        smoker_penalty = 5000 if input_features[4] == 1 else 0
        
        premium = base_premium + age_factor + bmi_factor + smoker_penalty
        return premium
    
    def give_suggestions(self, age, bmi, smoker, steps, exercise, sleep, marital_status):
        suggestions = []
        if smoker == 1:  
            suggestions.append("Quit smoking to reduce insurance premium.")
        if bmi > 30:
            suggestions.append("Maintain healthy diet and exercise to reduce BMI.")
        if age > 50:
            suggestions.append("Go for regular medical check-ups.")
        if steps < 5000:
            suggestions.append("Increase your daily steps to improve health.")
        if exercise < 2:
            suggestions.append("Do at least 2-3 hours of exercise per week.")
        if sleep < 6:
            suggestions.append("Increase your sleep hours for better recovery.")
        if marital_status.lower() == "single":
            suggestions.append("Consider family insurance plans if you marry in future.")
        
        suggestions.append("Maintaining good habits will reduce your premium over time.")
        return suggestions

predictor = LightweightPredictor()

@app.route('/')
def home():
    return """
    <html>
    <head>
        <title>Insurance Predictor</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body { font-family: Arial; margin: 20px; background: #f0f0f0; }
            .container { max-width: 500px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }
            input, select { width: 100%; padding: 10px; margin: 5px 0; border: 1px solid #ccc; border-radius: 5px; }
            button { background: #007bff; color: white; padding: 12px; border: none; border-radius: 5px; width: 100%; }
            .result { background: #e8f5e8; padding: 15px; margin: 10px 0; border-radius: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>🏥 Insurance Premium Predictor</h2>
            <form method="POST" action="/predict">
                <h3>Personal Info</h3>
                <input type="text" name="name" placeholder="Full Name" required>
                <input type="number" name="age" placeholder="Age" min="18" max="100" required>
                <select name="sex" required>
                    <option value="">Select Gender</option>
                    <option value="male">Male</option>
                    <option value="female">Female</option>
                </select>
                
                <h3>Health Info</h3>
                <input type="number" step="0.1" name="bmi" placeholder="BMI" min="15" max="50" required>
                <input type="number" name="children" placeholder="Children" min="0" max="10" required>
                <select name="smoker" required>
                    <option value="">Smoker?</option>
                    <option value="yes">Yes</option>
                    <option value="no">No</option>
                </select>
                
                <h3>Lifestyle</h3>
                <input type="number" name="daily_steps" placeholder="Daily Steps" required>
                <input type="number" step="0.5" name="exercise_hours" placeholder="Exercise hrs/week" required>
                <input type="number" step="0.5" name="sleep_hours" placeholder="Sleep hrs/day" required>
                <input type="number" name="salary" placeholder="Annual Salary" required>
                
                <button type="submit">Predict Premium</button>
            </form>
        </div>
    </body>
    </html>
    """

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        name = request.form['name']
        age = int(request.form['age'])
        sex = request.form['sex']
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        smoker = request.form['smoker']
        daily_steps = int(request.form['daily_steps'])
        exercise_hours = float(request.form['exercise_hours'])
        sleep_hours = float(request.form['sleep_hours'])
        salary = int(request.form['salary'])
        marital_status = "single"  # Default
        
        # Encode inputs
        try:
            sex_enc = predictor.le_dict['sex'].transform([sex])[0] if sex in predictor.le_dict['sex'].classes_ else 0
            smoker_enc = predictor.le_dict['smoker'].transform([smoker])[0] if smoker in predictor.le_dict['smoker'].classes_ else 0
            region_enc = 2  # Default southeast
            past_cond_enc = 0  # Default none
            profession_enc = 0  # Default
        except:
            # Fallback encoding
            sex_enc = 1 if sex == 'male' else 0
            smoker_enc = 1 if smoker == 'yes' else 0
            region_enc = 2
            past_cond_enc = 0
            profession_enc = 0
        
        # Create input array manually (without numpy)
        input_data = [[age, sex_enc, bmi, children, smoker_enc, past_cond_enc,
                      region_enc, profession_enc, salary, daily_steps,
                      exercise_hours, sleep_hours]]
        
        # Make prediction
        try:
            premium = predictor.model.predict(input_data)[0]
        except:
            premium = predictor.manual_predict(input_data[0])
        
        # Apply discounts
        discount = 0
        if smoker_enc == 0: discount += 0.05
        if bmi < 25: discount += 0.05
        if daily_steps >= 7000: discount += 0.05
        if exercise_hours >= 1.5: discount += 0.05
        if sleep_hours >= 7: discount += 0.05
        
        premium = premium * (1 - min(discount, 0.25))
        normal_insurance = premium * 0.6
        emergency_insurance = premium * 0.4
        
        # Generate suggestions
        suggestions = predictor.give_suggestions(age, bmi, smoker_enc, daily_steps, 
                                               exercise_hours, sleep_hours, marital_status)
        
        result_html = f"""
        <html>
        <head>
            <title>Result</title>
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body {{ font-family: Arial; margin: 20px; background: #f0f0f0; }}
                .container {{ max-width: 500px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }}
                .result {{ background: #e8f5e8; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .suggestion {{ background: #fff3cd; padding: 10px; margin: 5px 0; border-radius: 5px; }}
                button {{ background: #007bff; color: white; padding: 10px; border: none; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h2>📊 Result for {name}</h2>
                <div class="result">
                    <h3>💰 Premium: ₹{premium:.2f}</h3>
                    <p>📋 Normal (60%): ₹{normal_insurance:.2f}</p>
                    <p>🚨 Emergency (40%): ₹{emergency_insurance:.2f}</p>
                    <p>🎁 Discount: {min(discount * 100, 25):.1f}%</p>
                </div>
                
                <h3>💡 Suggestions:</h3>
                {"".join([f'<div class="suggestion">✓ {s}</div>' for s in suggestions])}
                
                <br>
                <button onclick="window.history.back()">← Back</button>
            </div>
        </body>
        </html>
        """
        
        return result_html
        
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    print("🚀 Starting lightweight insurance predictor...")
    print("📱 Access at: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
