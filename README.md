🏥 Medical Insurance Prediction System
A Flask web application that predicts medical insurance premiums using machine learning. The system analyzes user demographics, health metrics, and lifestyle factors to provide personalized insurance premium predictions with health improvement suggestions.

📋 Table of Contents
Features

Prerequisites

Installation

Project Structure

Usage

API Documentation

Model Details

Troubleshooting

✨ Features
🤖 Machine Learning Powered: Uses Random Forest Regressor for accurate premium predictions

💰 Premium Breakdown: Shows normal (60%) and emergency (40%) insurance components

🎁 Lifestyle Discounts: Automatically applies discounts for healthy habits (up to 25%)

💡 Personalized Suggestions: Provides health improvement recommendations

🌐 Web Interface: User-friendly form-based input system

🔌 REST API: Programmatic access to prediction functionality

📱 Responsive Design: Works on desktop and mobile devices

⚡ Real-time Predictions: Instant results with detailed breakdown

🛠 Prerequisites
Before running this application, ensure you have:

Python 3.8 or higher

The following files from your trained model:

insurance_model.pkl (trained Random Forest model)

encoders.pkl (fitted LabelEncoders for categorical variables)

📥 Installation
1. Clone or Download the Project
bash
# If using git
git clone https://github.com/rdineshd176-png/Medical-Insurance.git
cd Medical-Insurance.git

# Or simply download all files to a folder
2. Install Dependencies
bash
pip install -r requirements.txt
3. Place Model Files
Ensure these files are in the main project directory:

insurance_model.pkl

encoders.pkl

📁 Project Structure
text
insurance_app/
│
├── app.py                 # Main Flask application
├── insurance_model.pkl    # Trained ML model (required)
├── encoders.pkl          # Fitted encoders (required)
├── requirements.txt       # Python dependencies
│
├── templates/
│   ├── index.html        # Main input form
│   └── result.html       # Results display page
│
└── static/
    └── style.css         # CSS styling


🚀 Usage
Starting the Application
Open terminal/command prompt in the project directory

Run the Flask application:

bash
python app.py
Access the web application:
Open your browser and navigate to:

text
http://localhost:5000
Using the Web Interface
Fill out the form with your information:

Personal details (name, age, sex, marital status)

Health metrics (BMI, smoking status, medical history)

Lifestyle factors (daily steps, exercise, sleep)

Professional information (salary, profession, region)

Click "Predict Insurance Premium" to get your results

View your personalized report including:

Total predicted premium

Premium breakdown

Applied lifestyle discounts

Health improvement suggestions

Example Input
Field	Example Value
Name	John Doe
Age	35
Sex	Male
Marital Status	Married
BMI	24.5
Smoker	No
Daily Steps	8000
Exercise Hours	4
Sleep Hours	7
Salary	750000
Profession	Engineer
🔌 API Documentation
POST /api/predict
Make predictions programmatically via JSON API.

Endpoint: http://localhost:5000/api/predict
