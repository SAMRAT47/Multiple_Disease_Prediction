import streamlit as st
import pandas as pd
import numpy as np
import os
import base64
import joblib
from streamlit_option_menu import option_menu
import sys

# Import your prediction pipeline
from src.diseases.diabetes.pipeline.prediction_pipeline import PredictionPipeline
from src.diseases.heart.pipeline.prediction_pipeline import PredictionPipeline
from src.diseases.kidney.pipeline.prediction_pipeline import PredictionPipeline

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="Health Prediction Hub",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- STYLE -----------------

st.markdown("""
<style>
    /* Hide Streamlit's default UI elements */
    #MainMenu, footer, header {
        visibility: hidden;
    }
    
    /* General background and font styling */
    html, body, .stApp {
        color: #111 !important;
        font-family: 'Roboto', sans-serif;
    }
    
    h1, h2, h3 {
        color: #2c3e50 !important;
        font-weight: 700 !important;
    }
    
    h4, h5, h6, p, label {
        color: #34495e !important;
    }
    
    /* Card styling for content sections */
    .info-card {
        background-color: rgba(255, 255, 255, 0.85);
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    
    /* Form styling */
    .form-container {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        padding: 25px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Input fields */
    div[data-baseweb="input"] {
        background-color: #ffffff !important;
    }
    
    input[type="text"], input[type="number"] {
        background-color: white !important;
        color: #2c3e50 !important;
        border: none !important;
        padding: 10px !important;
        border-radius: 5px !important;
        border-bottom: 2px solid #3498db !important;
        transition: all 0.3s ease;
    }
    
    input[type="text"]:focus, input[type="number"]:focus {
        border-bottom: 2px solid #e74c3c !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05) !important;
    }
    
   /* ===== Dropdowns / Selectboxes ===== */
.stSelectbox div[data-baseweb="select"], 
.stSelectbox div[data-baseweb="select"] > div:first-child {
    background-color: #ffffff !important;
    color: #111 !important;
    border: 1px solid rgba(0, 0, 0, 0.1) !important;
    border-radius: 4px !important;
    font-size: 16px !important;
    padding: 8px !important;
}

.stSelectbox > div[data-baseweb="select"] > div:hover {
        background-color: #BEE3BA !important; /* Light blue hover effect */
        color: white !important; /* White text on hover */
}

.stSelectbox svg {
    fill: #111 !important;
    color: #111 !important;
}

/* ===== Dropdown Popover ===== */
div[data-baseweb="popover"] {
    background-color: #ffffff !important;
    border-radius: 5px !important;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
}

div[data-baseweb="popover"] * {
    background-color: white !important;
    color: black !important;
}

    /* Buttons */
    .stButton > button {
        background-color: #3498db !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 5px !important;
        padding: 10px 24px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1) !important;
    }
    
    .stButton > button:hover {
        background-color: #2980b9 !important;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Prediction result styling */
    .result-box {
        padding: 15px;
        border-radius: 5px;
        font-weight: 500;
        text-align: center;
        margin-top: 20px;
        font-size: 18px;
    }
    
    .result-positive {
        background-color: rgba(231, 76, 60, 0.2);
        color: #c0392b !important;
        border-left: 5px solid #e74c3c;
    }
    
    .result-negative {
        background-color: rgba(46, 204, 113, 0.2);
        color: #27ae60 !important;
        border-left: 5px solid #2ecc71;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: rgba(245, 245, 245, 0.9) !important;
        padding: 20px 10px !important;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background-color: #3498db !important;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- FUNCTIONS -----------------
def set_background(image_file):
    """Sets the background image for the page"""
    if os.path.exists(image_file):
        with open(image_file, "rb") as image:
            encoded = base64.b64encode(image.read()).decode()
        st.markdown(f"""
            <style>
            .stApp {{
                background-image: url("data:image/jpeg;base64,{encoded}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
            </style>
        """, unsafe_allow_html=True)
    else:
        # Use fallback gradient if image fails
        st.markdown("""
            <style>
            .stApp {
                background: linear-gradient(to right, #f5f7fa, #c3cfe2);
            }
            </style>
        """, unsafe_allow_html=True)

def set_sidebar_background(image_file):
    """Sets the background image for the sidebar"""
    if os.path.exists(image_file):
        with open(image_file, "rb") as image:
            encoded = base64.b64encode(image.read()).decode()
        st.markdown(f"""
            <style>
            [data-testid="stSidebar"] {{
                background-image: url("data:image/jpeg;base64,{encoded}");
                background-size: cover;
                background-position: center;
            }}
            </style>
        """, unsafe_allow_html=True)
    else:
        # Use fallback gradient if image fails
        st.markdown("""
            <style>
            [data-testid="stSidebar"] {
                background: linear-gradient(to bottom, #a1c4fd, #c2e9fb);
            }
            </style>
        """, unsafe_allow_html=True)

def display_banner(banner_path, alt_text="Banner"):
    """Displays a banner image if it exists, otherwise displays a styled header"""
    if os.path.exists(banner_path):
        st.image(banner_path,  use_container_width=True)
    else:
        st.markdown(f"""
            <div style="background: linear-gradient(to right, #3498db, #2c3e50); 
                       padding: 20px; border-radius: 10px; margin-bottom: 20px; 
                       text-align: center; color: white;">
                <h2 style="color: white !important;">{alt_text}</h2>
            </div>
        """, unsafe_allow_html=True)

def display_logo(logo_path):
    """Displays a logo image if it exists"""
    if os.path.exists(logo_path):
        st.sidebar.image(logo_path,  use_container_width=True)

def create_feature_input(label, key=None, options=None):
    """Creates appropriate input field based on whether options are provided"""
    if options:
        return st.selectbox(label, options=options, key=key)
    else:
        return st.number_input(label, value=0.0, step=0.1, format="%.2f", key=key)

def display_result(prediction_result, disease_name):
    """Displays prediction result with appropriate styling"""
    prediction = prediction_result.get("prediction", 0)
    message = prediction_result.get("message", "")
    
    if prediction == 1:
        st.markdown(f"""
            <div class="result-box result-positive">
                ⚠️ {message}
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="result-box result-negative">
                ✅ {message}
            </div>
        """, unsafe_allow_html=True)
    
    # Add additional information based on disease type
    if prediction == 1:
        if disease_name == "diabetes":
            st.info("Consider consulting with an endocrinologist for proper management of diabetes.")
        elif disease_name == "heart":
            st.info("Consider consulting with a cardiologist for further evaluation and heart health management.")
        elif disease_name == "kidney":
            st.info("Consider consulting with a nephrologist for further kidney function assessment.")
    else:
        st.success("Your health indicators look good for this condition. Continue maintaining a healthy lifestyle!")

# ----------------- INITIALIZE APP -----------------
# Create directories if they don't exist
os.makedirs("images", exist_ok=True)

# Define paths
image_dir = os.path.join(os.getcwd(), "images")
model_dir = os.path.join(os.getcwd(), "saved_models")
os.makedirs(model_dir, exist_ok=True)

# Define image paths
# You'll need to add your own images to these paths
home_bg = os.path.join(image_dir, "home_bg.jpg")
diabetes_bg = os.path.join(image_dir, "diabetes_bg.jpg")
heart_bg = os.path.join(image_dir, "heart_bg.jpg")
kidney_bg = os.path.join(image_dir, "kidney_bg.jpg")
sidebar_bg = os.path.join(image_dir, "sidebar_bg.jpg")
logo_path = os.path.join(image_dir, "logo.png")
home_banner = os.path.join(image_dir, "home_banner.jpg")
diabetes_banner = os.path.join(image_dir, "diabetes_banner.jpg")
heart_banner = os.path.join(image_dir, "heart_banner.jpg")
kidney_banner = os.path.join(image_dir, "kidney_banner.jpg")

# Initialize the prediction pipeline
try:
    prediction_pipeline = PredictionPipeline()
except Exception as e:
    st.error(f"Error initializing prediction pipeline: {e}")
    prediction_pipeline = None

# ----------------- SIDEBAR SETUP -----------------
set_sidebar_background(sidebar_bg)
display_logo(logo_path)

st.sidebar.markdown("<h2 style='text-align: center; color: #2c3e50;'>Health Prediction Hub</h2>", unsafe_allow_html=True)

# Create the sidebar menu with custom styling
with st.sidebar:
    menu_styles = {
        "container": {
            "padding": "0px",
            "background-color": "#BEE3BA",  # Soft green background
            "border-radius": "0px",
            "box-shadow": "none",           # Remove default shadow
            "border": "none"                # No border
        },
        "icon": {
            "color": "#333",
            "font-size": "20px"
        },
        "nav-link": {
            "font-size": "16px",
            "text-align": "left",
            "margin": "5px",
            "padding": "10px",
            "border-radius": "5px",
            "color": "#111",
            "background-color": "transparent",
            "box-shadow": "none",
            "border": "none"
        },
        "nav-link-selected": {
            "background-color": "#007BFF",
            "color": "white",
            "font-weight": "bold",
            "border-radius": "5px",
            "box-shadow": "none",
            "border": "none"
        },
        "menu-title": {
        "color": "#111111",  # Change this to any color you want, e.g., black
        "font-size": "20px",
        "font-weight": "bold",
        "text-align": "center",
        "padding": "10px 0"
        }
    }
    
    menu_icons = ["house-fill", "droplet-half", "heart-pulse", "kidneys"]
    
    selected = option_menu(
        "Disease Prediction",
        ["Home", "Diabetes Prediction", "Heart Disease Prediction", "Kidney Disease Prediction"],
        menu_icon="hospital-fill",
        icons=menu_icons,
        default_index=0,
        styles=menu_styles
    )
    
    # Add version info and credits
    st.sidebar.markdown("---")
    st.sidebar.caption("v1.0.0 | Health Prediction Hub")
    st.sidebar.caption("© 2025 | Your Company Name")

# ----------------- HOME PAGE -----------------
if selected == "Home":
    set_background(home_bg)
    display_banner(home_banner, "Welcome to Health Prediction Hub")
    
    # Welcome message
    st.markdown("""
        <div class="info-card">
            <h1 style="text-align: center;">Multiple Disease Prediction System 🏥</h1>
            <p style="font-size: 18px; text-align: center;">
                Early detection is key to better health outcomes. This system uses machine learning 
                to help predict potential health risks for diabetes, heart disease, and kidney disease.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Create three columns for disease info cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class="info-card" style="height: 100%;">
                <h3 style="text-align: center;">Diabetes Prediction 🧪</h3>
                <p>Our diabetes prediction model analyzes health parameters including:</p>
                <ul>
                    <li>Glucose levels</li>
                    <li>Body Mass Index (BMI)</li>
                    <li>Blood pressure</li>
                    <li>Insulin levels</li>
                    <li>Family history</li>
                </ul>
                <p>Early diabetes detection can help prevent complications like vision loss, kidney damage, and nerve problems.</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="info-card" style="height: 100%;">
                <h3 style="text-align: center;">Heart Disease Prediction 💓</h3>
                <p>Our heart disease prediction model evaluates:</p>
                <ul>
                    <li>Cholesterol levels</li>
                    <li>Blood pressure</li>
                    <li>Resting ECG results</li>
                    <li>Max heart rate</li>
                    <li>Chest pain patterns</li>
                </ul>
                <p>Heart disease remains the leading cause of death globally. Early detection can save lives.</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class="info-card" style="height: 100%;">
                <h3 style="text-align: center;">Kidney Disease Prediction 🔬</h3>
                <p>Our kidney disease prediction model examines:</p>
                <ul>
                    <li>Blood urea</li>
                    <li>Serum creatinine</li>
                    <li>Red blood cell count</li>
                    <li>White blood cell count</li>
                    <li>Albumin levels</li>
                </ul>
                <p>Chronic kidney disease often progresses silently. Early detection allows for interventions that can slow progression.</p>
            </div>
        """, unsafe_allow_html=True)
    
    # How to use section
    st.markdown("""
        <div class="info-card" style="margin-top: 30px;">
            <h2 style="text-align: center;">How to Use This Application</h2>
            <ol>
                <li><strong>Select a Disease Module:</strong> Choose from diabetes, heart disease, or kidney disease prediction in the sidebar.</li>
                <li><strong>Enter Your Health Data:</strong> Input your health parameters in the provided fields.</li>
                <li><strong>Get Your Prediction:</strong> Click the prediction button to see your results.</li>
                <li><strong>Consult a Professional:</strong> Remember that this tool provides predictions only. Always consult with healthcare professionals for proper diagnosis and treatment.</li>
            </ol>
        </div>
    """, unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
        <div class="info-card" style="background-color: rgba(231, 76, 60, 0.1); border-left: 5px solid #e74c3c;">
            <h3>Important Disclaimer</h3>
            <p>This prediction system is designed as a screening tool only and should not replace professional medical advice, diagnosis, or treatment. Always consult qualified healthcare providers with any questions regarding your medical condition.</p>
        </div>
    """, unsafe_allow_html=True)

# ----------------- DIABETES PREDICTION -----------------
elif selected == "Diabetes Prediction":
    set_background(diabetes_bg)
    display_banner(diabetes_banner, "Diabetes Prediction System")
    
    st.markdown("""
        <div class="info-card">
            <h1 style="text-align: center;">Diabetes Prediction 🧪</h1>
            <p style="text-align: center;">Enter your health parameters below to check your diabetes risk.</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Create input form
    st.markdown('<div class="form-container">', unsafe_allow_html=True)
    st.subheader("Enter Patient Information")
    
    # Create 3 columns for input fields
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
    with col2:
        glucose = st.number_input("Glucose (mg/dL)", min_value=0, max_value=300, value=70)
    with col3:
        blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70)
    
    with col1:
        skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
    with col2:
        insulin = st.number_input("Insulin (mu U/ml)", min_value=0, max_value=900, value=80)
    with col3:
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
    
    with col1:
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
    with col2:
        age = st.number_input("Age", min_value=0, max_value=120, value=35)
    with col3:
        st.markdown("<br><br>", unsafe_allow_html=True)
        predict_button = st.button("Predict Diabetes Risk")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Prediction logic
    if predict_button and prediction_pipeline:
        try:
            with st.spinner("Analyzing your data..."):
                # Prepare data dictionary for prediction
                diabetes_data = {
                    "Pregnancies": pregnancies,
                    "Glucose": glucose,
                    "BloodPressure": blood_pressure,
                    "SkinThickness": skin_thickness,
                    "Insulin": insulin,
                    "BMI": bmi,
                    "DiabetesPedigreeFunction": dpf,
                    "Age": age
                }
                
                # Make prediction
                result = prediction_pipeline.predict_diabetes(diabetes_data)
                
                # Display prediction
                display_result(result, "diabetes")
                
                # Display a chart showing important factors
                st.markdown("<h3>Key Risk Factors</h3>", unsafe_allow_html=True)
                importance_df = pd.DataFrame({
                    'Factor': ['Glucose', 'BMI', 'Age', 'Insulin', 'Blood Pressure'],
                    'Your Value': [glucose, bmi, age, insulin, blood_pressure],
                    'Reference Range': ['70-99 mg/dL', '18.5-24.9', 'N/A', '16-166 μIU/mL', '90/60-120/80 mmHg']
                })
                st.dataframe(importance_df, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error during prediction: {e}")
    
    # Health tips section
    st.markdown("""
        <div class="info-card" style="margin-top: 30px;">
            <h3>Diabetes Prevention Tips</h3>
            <ul>
                <li><strong>Maintain a healthy weight</strong> - Weight loss can prevent or delay diabetes</li>
                <li><strong>Stay physically active</strong> - Aim for 30 minutes of moderate activity 5 days a week</li>
                <li><strong>Eat healthy foods</strong> - Focus on fruits, vegetables, and whole grains</li>
                <li><strong>Control blood pressure and cholesterol</strong> - Get regular checkups</li>
                <li><strong>Quit smoking</strong> - Smoking increases your risk of diabetes and its complications</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

# ----------------- HEART DISEASE PREDICTION -----------------
elif selected == "Heart Disease Prediction":
    set_background(heart_bg)
    display_banner(heart_banner, "Heart Disease Prediction System")
    
    st.markdown("""
        <div class="info-card">
            <h1 style="text-align: center;">Heart Disease Prediction 💓</h1>
            <p style="text-align: center;">Enter your cardiac health parameters below to check your heart disease risk.</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Create input form
    st.markdown('<div class="form-container">', unsafe_allow_html=True)
    st.subheader("Enter Patient Information")
    
    # Create 3 columns for input fields
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=45)
    with col2:
        sex = st.selectbox("Sex", ["Male", "Female"])
        sex = 1 if sex == "Male" else 0
    with col3:
        cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], 
                          format_func=lambda x: {0: "Typical Angina", 1: "Atypical Angina", 
                                              2: "Non-anginal Pain", 3: "Asymptomatic"}[x])
    
    with col1:
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
    with col2:
        chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
    with col3:
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
        fbs = 1 if fbs == "Yes" else 0
    
    with col1:
        restecg = st.selectbox("Resting ECG", [0, 1, 2], 
                               format_func=lambda x: {0: "Normal", 1: "ST-T Wave Abnormality", 
                                                   2: "Left Ventricular Hypertrophy"}[x])
    with col2:
        thalach = st.number_input("Max Heart Rate", min_value=60, max_value=220, value=150)
    with col3:
        exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
        exang = 1 if exang == "Yes" else 0
    
    with col1:
        oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    with col2:
        slope = st.selectbox("Slope of Peak Exercise ST", [0, 1, 2], 
                             format_func=lambda x: {0: "Upsloping", 1: "Flat", 2: "Downsloping"}[x])
    with col3:
        ca = st.number_input("Number of Major Vessels Colored", min_value=0, max_value=4, value=0)
    
    with col1:
        thal = st.selectbox("Thalassemia", [1, 2, 3], 
                            format_func=lambda x: {1: "Normal", 2: "Fixed Defect", 3: "Reversible Defect"}[x])
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
    with col3:
        st.markdown("<br><br>", unsafe_allow_html=True)
        predict_button = st.button("Predict Heart Disease Risk")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Prediction logic
    if predict_button and prediction_pipeline:
        try:
            with st.spinner("Analyzing your cardiac data..."):
                # Prepare data dictionary for prediction
                heart_data = {
                    "age": age,
                    "sex": sex,
                    "cp": cp,
                    "trestbps": trestbps,
                    "chol": chol,
                    "fbs": fbs,
                    "restecg": restecg,
                    "thalach": thalach,
                    "exang": exang,
                    "oldpeak": oldpeak,
                    "slope": slope,
                    "ca": ca,
                    "thal": thal
                }
                
                # Make prediction
                result = prediction_pipeline.predict_heart_disease(heart_data)
                
                # Display prediction
                display_result(result, "heart")
                
                # Display key risk factors
                st.markdown("<h3>Key Cardiac Indicators</h3>", unsafe_allow_html=True)
                cardiac_df = pd.DataFrame({
                    'Factor': ['Cholesterol', 'Resting BP', 'Max Heart Rate', 'ST Depression', 'Major Vessels'],
                    'Your Value': [chol, trestbps, thalach, oldpeak, ca],
                    'Optimal Range': ['< 200 mg/dL', '< 120/80 mmHg', '(220 - age) * 0.85', '< 0.5', '0']
                })
                st.dataframe(cardiac_df, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error during prediction: {e}")
    
    # Health tips section
    st.markdown("""
        <div class="info-card" style="margin-top: 30px;">
            <h3>Heart Health Tips</h3>
            <ul>
                <li><strong>Exercise regularly</strong> - At least 150 minutes of moderate activity weekly</li>
                <li><strong>Eat a heart-healthy diet</strong> - Rich in fruits, vegetables, whole grains, and lean proteins</li>
                <li><strong>Maintain healthy cholesterol levels</strong> - Keep LDL below 100 mg/dL</li>
                <li><strong>Control blood pressure</strong> - Aim for below 120/80 mmHg</li>
                <li><strong>Manage stress</strong> - Practice relaxation techniques like meditation</li>
                <li><strong>Get quality sleep</strong> - Aim for 7-8 hours per night</li>
                <li><strong>Avoid tobacco</strong> - Smoking damages blood vessels and heart</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

# ----------------- KIDNEY DISEASE PREDICTION -----------------
elif selected == "Kidney Disease Prediction":
    set_background(kidney_bg)
    display_banner(kidney_banner, "Kidney Disease Prediction System")
    
    st.markdown("""
        <div class="info-card">
            <h1 style="text-align: center;">Kidney Disease Prediction 🔬</h1>
            <p style="text-align: center;">Enter your kidney health parameters below to check your risk of kidney disease.</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Create input form in tabs to manage many inputs
    st.markdown('<div class="form-container">', unsafe_allow_html=True)
    
    # Create tabs for different categories of inputs
    tab1, tab2, tab3 = st.tabs(["Blood Tests", "Urine Tests", "Medical History"])
    
    # Dictionary to store all inputs
    kidney_data = {}
    
    # Tab 1: Blood Tests
    with tab1:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            kidney_data["age"] = st.number_input("Age", min_value=1, max_value=120, value=45, key="k_age")
        with col2:
            kidney_data["blood_pressure"] = st.number_input("Blood Pressure (mm Hg)", min_value=50, max_value=200, value=80, key="k_bp")
        with col3:
            kidney_data["blood_glucose_random"] = st.number_input("Blood Glucose Random (mg/dL)", min_value=0, max_value=500, value=100, key="k_bgr")
            
        with col1:
            kidney_data["blood_urea"] = st.number_input("Blood Urea (mg/dL)", min_value=0, max_value=200, value=25, key="k_bu")
        with col2:
            kidney_data["serum_creatinine"] = st.number_input("Serum Creatinine (mg/dL)", min_value=0.0, max_value=15.0, value=1.0, step=0.1, key="k_sc")
        with col3:
            kidney_data["sodium"] = st.number_input("Sodium (mEq/L)", min_value=100, max_value=200, value=135, key="k_sod")
            
        with col1:
            kidney_data["potassium"] = st.number_input("Potassium (mEq/L)", min_value=2.0, max_value=8.0, value=4.0, step=0.1, key="k_pot")
                                                       
        with col2:
            kidney_data["hemoglobin"] = st.number_input("Hemoglobin (g/dL)", min_value=0.0, max_value=20.0, value=13.0, step=0.1, key="k_hemo")
        with col3:
            kidney_data["packed_cell_volume"] = st.number_input("Packed Cell Volume", min_value=0, max_value=60, value=40, key="k_pcv")
            
        with col1:
            kidney_data["white_blood_cell_count"] = st.number_input("White Blood Cell Count (cells/mm³)", min_value=0, max_value=50000, value=9000, key="k_wc")
        with col2:
            kidney_data["red_blood_cell_count"] = st.number_input("Red Blood Cell Count (millions/mm³)", min_value=0.0, max_value=10.0, value=4.5, step=0.1, key="k_rc")
        
    # Tab 2: Urine Tests
    with tab2:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            kidney_data["albumin"] = st.selectbox("Albumin in Urine", [0, 1, 2, 3, 4, 5], 
                                              format_func=lambda x: {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5"}[x], key="k_al")
        with col2:
            kidney_data["sugar"] = st.selectbox("Sugar in Urine", [0, 1, 2, 3, 4, 5], 
                                           format_func=lambda x: {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5"}[x], key="k_su")
        with col3:
            kidney_data["red_blood_cells"] = st.selectbox("Red Blood Cells in Urine", ["Normal", "Abnormal"], key="k_rbc")
            kidney_data["red_blood_cells"] = 1 if kidney_data["red_blood_cells"] == "Abnormal" else 0
            
        with col1:
            kidney_data["pus_cell"] = st.selectbox("Pus Cell", ["Normal", "Abnormal"], key="k_pc")
            kidney_data["pus_cell"] = 1 if kidney_data["pus_cell"] == "Abnormal" else 0
        with col2:
            kidney_data["pus_cell_clumps"] = st.selectbox("Pus Cell Clumps", ["Not Present", "Present"], key="k_pcc")
            kidney_data["pus_cell_clumps"] = 1 if kidney_data["pus_cell_clumps"] == "Present" else 0
        with col3:
            kidney_data["bacteria"] = st.selectbox("Bacteria", ["Not Present", "Present"], key="k_ba")
            kidney_data["bacteria"] = 1 if kidney_data["bacteria"] == "Present" else 0
            
    # Tab 3: Medical History
    with tab3:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            kidney_data["hypertension"] = st.selectbox("Hypertension", ["No", "Yes"], key="k_htn")
            kidney_data["hypertension"] = 1 if kidney_data["hypertension"] == "Yes" else 0
        with col2:
            kidney_data["diabetes_mellitus"] = st.selectbox("Diabetes Mellitus", ["No", "Yes"], key="k_dm")
            kidney_data["diabetes_mellitus"] = 1 if kidney_data["diabetes_mellitus"] == "Yes" else 0
        with col3:
            kidney_data["coronary_artery_disease"] = st.selectbox("Coronary Artery Disease", ["No", "Yes"], key="k_cad")
            kidney_data["coronary_artery_disease"] = 1 if kidney_data["coronary_artery_disease"] == "Yes" else 0
            
        with col1:
            kidney_data["appetite"] = st.selectbox("Appetite", ["Good", "Poor"], key="k_appet")
            kidney_data["appetite"] = 1 if kidney_data["appetite"] == "Poor" else 0
        with col2:
            kidney_data["pedal_edema"] = st.selectbox("Pedal Edema", ["No", "Yes"], key="k_pe")
            kidney_data["pedal_edema"] = 1 if kidney_data["pedal_edema"] == "Yes" else 0
        with col3:
            kidney_data["anemia"] = st.selectbox("Anemia", ["No", "Yes"], key="k_ane")
            kidney_data["anemia"] = 1 if kidney_data["anemia"] == "Yes" else 0
    
    # Submit button - outside the tabs
    predict_button = st.button("Predict Kidney Disease Risk")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Prediction logic
    if predict_button and prediction_pipeline:
        try:
            with st.spinner("Analyzing your kidney data..."):
                # Make prediction
                result = prediction_pipeline.predict_kidney_disease(kidney_data)
                
                # Display prediction
                display_result(result, "kidney")
                
                # Display key risk factors
                st.markdown("<h3>Key Kidney Health Indicators</h3>", unsafe_allow_html=True)
                kidney_df = pd.DataFrame({
                    'Test': ['Blood Urea', 'Serum Creatinine', 'Hemoglobin', 'Red Blood Cell Count', 'Albumin'],
                    'Your Value': [
                        kidney_data["blood_urea"],
                        kidney_data["serum_creatinine"],
                        kidney_data["hemoglobin"],
                        kidney_data["red_blood_cell_count"],
                        kidney_data["albumin"]
                    ],
                    'Reference Range': ['7-20 mg/dL', '0.5-1.5 mg/dL', '13.5-17.5 g/dL', '4.5-5.5 million/mm³', '0']
                })
                st.dataframe(kidney_df, use_container_width=True)
                
                # Create eGFR calculation (estimated Glomerular Filtration Rate)
                # Using the CKD-EPI equation
                gender = "male"  # Hardcoded for simplicity
                age = kidney_data["age"]
                scr = kidney_data["serum_creatinine"]
                
                if gender == "female":
                    if scr <= 0.7:
                        egfr = 144 * pow((scr / 0.7), -0.329) * pow(0.993, age)
                    else:
                        egfr = 144 * pow((scr / 0.7), -1.209) * pow(0.993, age)
                else:
                    if scr <= 0.9:
                        egfr = 141 * pow((scr / 0.9), -0.411) * pow(0.993, age)
                    else:
                        egfr = 141 * pow((scr / 0.9), -1.209) * pow(0.993, age)
                
                st.info(f"Estimated GFR: {egfr:.2f} mL/min/1.73m²")
                
                # CKD Stage information
                if egfr >= 90:
                    st.success("Normal kidney function (eGFR ≥ 90)")
                elif egfr >= 60:
                    st.warning("Mildly reduced kidney function (eGFR 60-89)")
                elif egfr >= 45:
                    st.warning("Mild to moderate reduction in kidney function (eGFR 45-59)")
                elif egfr >= 30:
                    st.error("Moderate to severe reduction in kidney function (eGFR 30-44)")
                elif egfr >= 15:
                    st.error("Severe reduction in kidney function (eGFR 15-29)")
                else:
                    st.error("Kidney failure (eGFR < 15)")
                
        except Exception as e:
            st.error(f"Error during prediction: {e}")
    
    # Health tips section
    st.markdown("""
        <div class="info-card" style="margin-top: 30px;">
            <h3>Kidney Health Tips</h3>
            <ul>
                <li><strong>Stay hydrated</strong> - Drink 1.5 to 2 liters of water daily</li>
                <li><strong>Maintain a healthy blood pressure</strong> - High blood pressure can damage kidneys</li>
                <li><strong>Control blood sugar</strong> - Diabetes is a leading cause of kidney disease</li>
                <li><strong>Reduce salt intake</strong> - Aim for less than 5g per day</li>
                <li><strong>Eat a balanced diet</strong> - Rich in fruits, vegetables, and whole grains</li>
                <li><strong>Exercise regularly</strong> - At least 30 minutes of moderate activity 5 days a week</li>
                <li><strong>Avoid NSAIDs</strong> - Long-term use can harm kidney function</li>
                <li><strong>Quit smoking</strong> - Smoking reduces blood flow to kidneys</li>
                <li><strong>Get regular checkups</strong> - Monitor kidney function with blood and urine tests</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

# ----------------- Create the Prediction Pipeline Class -----------------
# This would normally be in a separate file, but including it here for completeness

class PredictionPipeline:
    """
    Class for making predictions for different diseases.
    In a real implementation, this would load trained models from disk.
    """
    
    def __init__(self):
        """Initialize the prediction pipeline by loading models."""
        self.models = {}
        self.load_models()
    
    def load_models(self):
        """Load all prediction models from disk."""
        model_dir = os.path.join(os.getcwd(), "saved_models")
        os.makedirs(model_dir, exist_ok=True)
        
        # For demonstration purposes, we'll create dummy models if they don't exist
        # In a real application, you'd load your trained models
        
        # Diabetes model
        diabetes_model_path = os.path.join(model_dir, "diabetes_model.pkl")
        if os.path.exists(diabetes_model_path):
            self.models["diabetes"] = joblib.load(diabetes_model_path)
        else:
            # Create a dummy model for demonstration
            from sklearn.ensemble import RandomForestClassifier
            self.models["diabetes"] = RandomForestClassifier(n_estimators=10, random_state=42)
        
        # Heart disease model
        heart_model_path = os.path.join(model_dir, "heart_model.pkl")
        if os.path.exists(heart_model_path):
            self.models["heart"] = joblib.load(heart_model_path)
        else:
            # Create a dummy model for demonstration
            from sklearn.ensemble import RandomForestClassifier
            self.models["heart"] = RandomForestClassifier(n_estimators=10, random_state=42)
        
        # Kidney disease model
        kidney_model_path = os.path.join(model_dir, "kidney_model.pkl")
        if os.path.exists(kidney_model_path):
            self.models["kidney"] = joblib.load(kidney_model_path)
        else:
            # Create a dummy model for demonstration
            from sklearn.ensemble import RandomForestClassifier
            self.models["kidney"] = RandomForestClassifier(n_estimators=10, random_state=42)
    
    def predict_diabetes(self, data):
        """Make diabetes prediction based on input data."""
        # In a real implementation, you would:
        # 1. Preprocess the data
        # 2. Make prediction using the model
        # 3. Return the result
        
        # For demonstration, we'll create a simple rule-based prediction
        glucose = data["Glucose"]
        bmi = data["BMI"]
        age = data["Age"]
        
        # Simple rule-based prediction for demonstration
        if glucose > 140 and bmi > 30:
            prediction = 1
            message = "High risk of diabetes. Your glucose and BMI values are elevated."
        elif glucose > 140 or bmi > 30:
            prediction = 1
            message = "Moderate risk of diabetes. Please consult with a healthcare professional."
        else:
            prediction = 0
            message = "Low risk of diabetes. Your parameters appear normal."
        
        return {"prediction": prediction, "message": message}
    
    def predict_heart_disease(self, data):
        """Make heart disease prediction based on input data."""
        # Similar to diabetes, for demonstration we'll use a rule-based approach
        
        age = data["age"]
        cholesterol = data["chol"]
        thalach = data["thalach"] # max heart rate
        cp = data["cp"] # chest pain type
        ca = data["ca"] # number of major vessels colored
        
        # Simple rule-based prediction for demonstration
        if age > 60 and cholesterol > 240 and ca >= 1:
            prediction = 1
            message = "High risk of heart disease. Multiple risk factors detected."
        elif (age > 55 and cholesterol > 200) or cp == 3 or ca >= 1:
            prediction = 1
            message = "Moderate risk of heart disease. Please consult with a cardiologist."
        else:
            prediction = 0
            message = "Low risk of heart disease. Your cardiac parameters appear normal."
        
        return {"prediction": prediction, "message": message}
    
    def predict_kidney_disease(self, data):
        """Make kidney disease prediction based on input data."""
        # Similar to above, using a rule-based approach for demonstration
        
        blood_urea = data["blood_urea"]
        serum_creatinine = data["serum_creatinine"]
        albumin = data["albumin"]
        hypertension = data["hypertension"]
        diabetes_mellitus = data["diabetes_mellitus"]
        
        # Simple rule-based prediction for demonstration
        if (serum_creatinine > 1.5 and blood_urea > 40) or albumin >= 3:
            prediction = 1
            message = "High risk of kidney disease. Several abnormal kidney function indicators."
        elif serum_creatinine > 1.4 or blood_urea > 30 or albumin > 0 or (hypertension and diabetes_mellitus):
            prediction = 1
            message = "Moderate risk of kidney disease. Please consult with a nephrologist."
        else:
            prediction = 0
            message = "Low risk of kidney disease. Your kidney parameters appear normal."
        
        return {"prediction": prediction, "message": message}


# Run the app
if __name__ == "__main__":
    st.write("Running Health Prediction Hub application")