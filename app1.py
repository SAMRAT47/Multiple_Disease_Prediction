
import streamlit as st
import joblib
import os
import base64
from streamlit_option_menu import option_menu

# ----------------- PAGE CONFIG -----------------
st.set_page_config(page_title="Multiple Disease Prediction", layout="wide", page_icon="🩺")

# ----------------- STYLE -----------------

st.markdown("""
<style>
    /* Hide Streamlit's default UI elements */
    #MainMenu, footer, header {
        visibility: hidden;
    }
    
    /* General background and font styling */
    html, body, .stApp {
        background-color: #f7f7f7;
        color: #111 !important;
        font-family: 'Segoe UI', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6, p, label {
        color: #111 !important;
    }
    
    /* Fix input fields */
    div[data-baseweb="input"] {
        background-color: #ffffff !important;
        border: none !important;
        box-shadow: none !important;
    }
    
    input[type="text"], input[type="number"] {
        pointer-events: auto !important;
        z-index: auto !important;
        position: relative !important;
        background-color: white !important;
        color: black !important;
        border: none !important;
        padding: 10px !important;
        outline: none !important;
        box-shadow: none !important;
        border-bottom: 2px solid #007BFF !important;  /* Default line */
        caret-color: #007BFF !important;  /* Cursor color */
    }
    
    input[type="text"]:focus, input[type="number"]:focus {
        outline: none !important;
        box-shadow: none !important;
        border: none !important;
        border-bottom: 2px solid #FF5733 !important;  /* Highlighted line on focus */
        caret-color: #FF5733 !important;  /* Cursor color on focus */
    }
    
    /* Ensure inputs are editable */
    input, textarea {
        z-index: auto !important;
        position: static !important;
        pointer-events: auto !important;
    }
    
    /* DROPDOWN SELECTBOX */
    .stSelectbox div[data-baseweb="select"], .stSelectbox div[data-baseweb="select"] > div:first-child {
        background-color: #ffffff !important;
        color: #111 !important;
        border: 1px solid rgba(0,0,0,0.1) !important;
        border-radius: 4px !important;
        font-size: 16px !important;
    }
    
    /* Popover for dropdown options */
    div[data-baseweb="popover"] {
        background-color: #ffffff !important;
    }
    
    div[data-baseweb="popover"] * {
        background-color: white !important;
        color: black !important;
    }
    
    /* BUTTONS */
    .stButton > button {
        background-color: #007BFF !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: bold !important;
        padding: 10px 20px;
        font-size: 16px;
    }
    
    .stButton > button:hover {
        background-color: #0056b3 !important;
    }
    
    /* SIDEBAR CLEANUP */
    [data-testid="stSidebar"] {
        background-color: #f5f5f5 !important;
        border: none !important;
        box-shadow: none !important;
        padding: 1rem;
    }
    
    [data-testid="stSidebar"] * {
        color: black !important;
        background: transparent !important;
    }
    
    /* Remove Streamlit default dark corners */
    .css-1d391kg, .css-1v3fvcr, .css-1d0zn9t {
        background: transparent !important;
    }
</style>
""", unsafe_allow_html=True)



# ----------------- FUNCTIONS -----------------
def set_background(image_file):
    """
    Sets the background image for the page.
    Checks if the file exists before attempting to load it.
    """
    # Check if the file exists and path is valid
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
        # Use fallback color instead of image
        st.markdown("""
            <style>
            .stApp {
                background-color: #f5f5f5 !important;
            }
            </style>
        """, unsafe_allow_html=True)

def set_sidebar_background(image_file):
    """
    Sets the background image for the sidebar.
    Checks if the file exists before attempting to load it.
    """
    # Check if the file exists and path is valid
    if os.path.exists(image_file):
        with open(image_file, "rb") as image:
            encoded = base64.b64encode(image.read()).decode()
        st.markdown(f"""
            <style>
            [data-testid="stSidebar"] {{
                background-image: url("data:image/jpeg;base64,{encoded}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                border-radius: 20px;
                padding: 10px;
            }}
            </style>
        """, unsafe_allow_html=True)
    else:
        # Use fallback gradient if image fails
        st.markdown("""
            <style>
            [data-testid="stSidebar"] {
                background-image: linear-gradient(to bottom, #a8edea, #fed6e3);
                border-radius: 20px;
                padding: 10px;
            }
            </style>
        """, unsafe_allow_html=True)


# ----------------- MODEL LOADER -----------------
def load_model_with_version(path, model_name=""):
    try:
        loaded = joblib.load(path)
        if isinstance(loaded, tuple) and len(loaded) == 2:
            model, version = loaded
            return model
        else:
            return loaded  # Fallback in case version wasn't saved
    except Exception as e:
        st.error(f"Error loading {model_name} model: {e}")
        st.stop()

# ----------------- LOAD MODELS -----------------
# Define paths with more robust path joining
current_dir = os.getcwd()
model_dir = os.path.join(current_dir, "notebooks", "saved_models")
image_dir = os.path.join(current_dir, "images")

# Debug image directory
if not os.path.exists(image_dir):
    # Try to create the directory if needed
    os.makedirs(image_dir, exist_ok=True)

# Load the trained machine learning models
diabetes_model = load_model_with_version(os.path.join(model_dir, "diabetes_model.pkl"), "Diabetes")
heart_disease_model = load_model_with_version(os.path.join(model_dir, "heart_model.pkl"), "Heart Disease")
kidney_disease_model = load_model_with_version(os.path.join(model_dir, "kidney_model.pkl"), "Kidney Disease")

# -------- SIDEBAR CUSTOMIZATION --------
# Set the sidebar background first
sidebar_bg_path = os.path.join(image_dir, "sidebar_bg.jpg")
set_sidebar_background(sidebar_bg_path)

# Add sidebar logo if available
sidebar_logo_path = os.path.join(image_dir, "sidebar_logo.png")
if os.path.exists(sidebar_logo_path):
    st.sidebar.image(sidebar_logo_path, use_column_width=True)

# Add custom title to sidebar
st.sidebar.markdown("<h2 style='text-align: center;'>Health Prediction Hub</h2>", unsafe_allow_html=True)

# Customizable sidebar menu
with st.sidebar:
    # Custom styles for option menu
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
    
    # Custom icons - using more relevant icons for each disease
    menu_icons = ['house-fill', 'droplet-half', 'heart-pulse', 'kidneys']
    
    # Create the option menu with custom styles
    selected = option_menu(
        "Disease Prediction",
        ["Home", "Diabetes Prediction", "Heart Disease Prediction", "Kidney Disease Prediction"],
        menu_icon='hospital-fill',
        icons=menu_icons,
        default_index=0,
        styles=menu_styles
    )
    

# ----------------- HOME SECTION -----------------
if selected == "Home":
    # Use a default color if background image not available
    set_background(os.path.join(image_dir, "home_bg.jpg"))
        
    # Show banner if available, otherwise display title only
    banner_path = os.path.join(image_dir, "home_banner.jpg")
    if os.path.exists(banner_path):
        st.image(banner_path, use_container_width=True)
        
    st.markdown("""
        <div style="text-align: center; padding: 20px; border-radius: 10px; background-color: rgba(255,255,255,0.8); margin-bottom: 20px;">
            <h1>Welcome to the Multi-Disease Prediction System 🏥</h1>
            <p style="font-size: 20px;">This system helps you predict potential health issues like Diabetes, Heart Disease, and Kidney Disease using Machine Learning models.</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Add instructions
    st.markdown("""
        <div style="padding: 20px; border-radius: 10px; background-color: rgba(255,255,255,0.8);">
            <h2>How to use this application:</h2>
            <ol>
                <li>Select a disease prediction module from the sidebar menu</li>
                <li>Enter your health data in the provided fields</li>
                <li>Click the prediction button to see your results</li>
            </ol>
            <p>This application provides predictions for three common health conditions:</p>
            <ul>
                <li><strong>Diabetes</strong> - Based on factors like glucose levels, BMI, and other metrics</li>
                <li><strong>Heart Disease</strong> - Based on cardiac health indicators and risk factors</li>
                <li><strong>Kidney Disease</strong> - Based on blood tests and other kidney health markers</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

# ----------------- DIABETES SECTION -----------------
if selected == "Diabetes Prediction":
    # Set background
    set_background(os.path.join(image_dir, "diabetes_bg.jpg"))
    
    # Show banner if available    
    banner_path = os.path.join(image_dir, "diabetes_banner.jpg")
    if os.path.exists(banner_path):
        st.image(banner_path, use_container_width=True)
    
    st.markdown("""
        <div style="text-align: center; padding: 15px; border-radius: 10px; background-color: rgba(255,255,255,0.8); margin-bottom: 20px;">
            <h1>🧪 Diabetes Prediction Using Machine Learning</h1>
        </div>
    """, unsafe_allow_html=True)

    # Create form container with background
    st.markdown("""
        <div style="padding: 20px; border-radius: 10px; background-color: rgba(255,255,255,0.8); margin-bottom: 20px;">
            <h3>Enter Patient Information</h3>
        </div>
    """, unsafe_allow_html=True)

    # Create a form with 3 columns for input fields
    col1, col2, col3 = st.columns(3)
    with col1: Pregnancies = st.text_input("Pregnancies")
    with col2: Glucose = st.text_input("Glucose")
    with col3: BloodPressure = st.text_input("Blood Pressure")
    with col1: SkinThickness = st.text_input("Skin Thickness")
    with col2: Insulin = st.text_input("Insulin")
    with col3: BMI = st.text_input("BMI")
    with col1: DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function")
    with col2: Age = st.text_input("Age")

    # Prediction button and logic
    if st.button("Diabetes Test Result"):
        try:
            # Process input and convert to appropriate format
            bmi = float(BMI) if BMI else 0
            glucose = float(Glucose) if Glucose else 0
            insulin = float(Insulin) if Insulin else 0
            user_input = [
                float(Pregnancies) if Pregnancies else 0, 
                float(Glucose) if Glucose else 0, 
                float(BloodPressure) if BloodPressure else 0, 
                float(SkinThickness) if SkinThickness else 0, 
                float(Insulin) if Insulin else 0,
                float(BMI) if BMI else 0, 
                float(DiabetesPedigreeFunction) if DiabetesPedigreeFunction else 0, 
                float(Age) if Age else 0,
                int(bmi <= 18.5), int(24.9 < bmi <= 29.9), int(29.9 < bmi <= 34.9),
                int(34.9 < bmi <= 39.9), int(bmi > 39.9),
                int(16 <= insulin <= 166),
                int(glucose <= 70), int(70 < glucose <= 99),
                int(99 < glucose <= 126), int(glucose > 126)
            ]
            prediction = diabetes_model.predict([user_input])
            result = "✅ The person is diabetic" if prediction[0] == 1 else "🟢 The person is not diabetic"
            st.success(result)
        except Exception as e:
            st.error(f"Input error: {e}")

# ----------------- HEART SECTION -----------------
if selected == "Heart Disease Prediction":
    # Set background
    set_background(os.path.join(image_dir, "heart_bg.jpg"))
    
    # Show banner if available    
    banner_path = os.path.join(image_dir, "heart_banner.jpg")
    if os.path.exists(banner_path):
        st.image(banner_path, use_container_width=True)
        
    st.markdown("""
        <div style="text-align: center; padding: 15px; border-radius: 10px; background-color: rgba(255,255,255,0.8); margin-bottom: 20px;">
            <h1>💓 Heart Disease Prediction</h1>
        </div>
    """, unsafe_allow_html=True)

    # Create form container with background
    st.markdown("""
        <div style="padding: 20px; border-radius: 10px; background-color: rgba(255,255,255,0.8); margin-bottom: 20px;">
            <h3>Enter Patient Information</h3>
        </div>
    """, unsafe_allow_html=True)

    # Create a form with 3 columns for input fields
    col1, col2, col3 = st.columns(3)
    with col1: age = st.text_input("Age")
    with col2:
        sex = st.selectbox("Sex", ["Male", "Female"])
        sex = "1" if "Male" in sex else "0"
    with col3: cp = st.text_input("Chest Pain Type")
    with col1: trestbps = st.text_input("Resting Blood Pressure")
    with col2: chol = st.text_input("Serum Cholesterol")
    with col3:
        fbs = st.selectbox("Fasting Blood Sugar > 120", ["True", "False"])
        fbs = "1" if "True" in fbs else "0"
    with col1: restecg = st.text_input("Rest ECG")
    with col2: thalach = st.text_input("Max Heart Rate")
    with col3:
        exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
        exang = "1" if "Yes" in exang else "0"
    with col1: oldpeak = st.text_input("ST Depression")
    with col2: slope = st.text_input("Slope")
    with col3: ca = st.text_input("Major Vessels Colored")
    with col1:
        thal = st.selectbox("Thal", ["Normal", "Fixed defect", "Reversible defect"])
        if "Normal" in thal:
            thal = "3"
        elif "Fixed" in thal:
            thal = "6"
        else:
            thal = "7"

       # Add extra styling fix for dropdown menus
    st.markdown("""
    <style>
    /* This is a more aggressive fix for dropdowns */
    div[data-baseweb="popover"] * {
    background-color: white !important;
    color: black !important;
    font-weight: normal !important;
    }

    div[data-baseweb="menu"] * {
    background-color: white !important;
    color: black !important;
    font-weight: normal !important;
    }

    /* Fix option font color */
    div[role="listbox"] ul li, div[role="option"] {
    color: black !important;
    background: white !important;
    font-weight: normal !important;
    }

    /* Make dropdown arrow visible */
    div[data-baseweb="select"] svg {
    color: black !important;
    visibility: visible !important;
    display: inline-block !important;
    opacity: 1 !important;
    }

    </style>
    """, unsafe_allow_html=True)

    # Prediction button and logic
    if st.button("Heart Disease Test Result"):
        try:
            # Process input and make prediction
            user_input = [
                float(age) if age else 0,
                float(sex),
                float(cp) if cp else 0,
                float(trestbps) if trestbps else 0,
                float(chol) if chol else 0,
                float(fbs),
                float(restecg) if restecg else 0,
                float(thalach) if thalach else 0,
                float(exang),
                float(oldpeak) if oldpeak else 0,
                float(slope) if slope else 0,
                float(ca) if ca else 0,
                float(thal)
            ]
            prediction = heart_disease_model.predict([user_input])
            result = "❗ The person has heart disease" if prediction[0] == 1 else "🟢 The person does not have heart disease"
            st.success(result)
        except Exception as e:
            st.error(f"Input error: {e}")

# ----------------- KIDNEY SECTION -----------------
if selected == "Kidney Disease Prediction":
    # Set background with fallback
    try:
        set_background(os.path.join(image_dir, "kidney_bg.jpg"))
    except:
        st.markdown("""
            <style>
            .stApp {
                background-color: #f0fff5 !important;
            }
            </style>
        """, unsafe_allow_html=True)
    
    # Show banner if available    
    banner_path = os.path.join(image_dir, "kidney_banner.jpg")
    if os.path.exists(banner_path):
        st.image(banner_path, use_container_width=True)
        
    st.markdown("""
        <div style="text-align: center; padding: 15px; border-radius: 10px; background-color: rgba(255,255,255,0.8); margin-bottom: 20px;">
            <h1>🔬 Kidney Disease Prediction</h1>
        </div>
    """, unsafe_allow_html=True)


        # Create form container with background
    st.markdown("""
        <div style="padding: 20px; border-radius: 10px; background-color: rgba(255,255,255,0.8); margin-bottom: 20px;">
            <h3>Enter Patient Information</h3>
        </div>
    """, unsafe_allow_html=True)

    # Define categorical fields and their options
    categorical_fields = {
        "Red Blood Cell": ["Normal", "Abnormal"],
        "Pus Cell": ["Normal", "Abnormal"],
        "Pus Cell Clumps": ["Present", "Not Present"],
        "Bacteria": ["Present", "Not Present"],
        "Hypertension": ["Yes", "No"],
        "Diabetes Mellitus": ["Yes", "No"],
        "Coronary Artery Disease": ["Yes", "No"],
        "Appetite": ["Good", "Poor"],
        "Peda Edema": ["Yes", "No"],
        "Anemia": ["Yes", "No"]
    }

    # Store all user inputs
    user_inputs = {}
    cols = st.columns(5)

    # Create a form with 5 columns for input fields
    idx = 0
    for field in ["Age", "Blood Pressure", "Specific Gravity", "Albumin", "Sugar",
                  "Blood Glucose Random", "Blood Urea", "Serum Creatinine", "Sodium",
                  "Potassium", "Haemoglobin", "Packed Cell Volume",
                  "White Blood Cell Count", "Red Blood Cell Count"] + list(categorical_fields.keys()):
        with cols[idx % 5]:
            if field in categorical_fields:
                # Use selectbox for categorical fields with custom style
                key = f"select_{field.replace(' ', '_').lower()}"
                user_inputs[field] = st.selectbox(
                    field, 
                    categorical_fields[field],
                    key=key
                )
                # Add custom style for this specific selectbox
                st.markdown(f"""
                    <style>
                    div[data-testid="stSelectbox"][aria-labelledby="select_{field.replace(' ', '_').lower()}-label"] div[data-baseweb="select"] div {{
                        background-color: white !important;
                        color: #111 !important;
                    }}
                    </style>
                """, unsafe_allow_html=True)
            else:
                # Use text_input for numerical fields
                user_inputs[field] = st.text_input(field)
        idx += 1

    # Add extra styling fix for dropdown menus
    st.markdown("""
    <style>
    /* This is a more aggressive fix for dropdowns */
    div[data-baseweb="popover"] * {
    background-color: white !important;
    color: black !important;
    font-weight: normal !important;
    }

    div[data-baseweb="menu"] * {
    background-color: white !important;
    color: black !important;
    font-weight: normal !important;
    }

    /* Fix option font color */
    div[role="listbox"] ul li, div[role="option"] {
    color: black !important;
    background: white !important;
    font-weight: normal !important;
    }

    /* Make dropdown arrow visible */
    div[data-baseweb="select"] svg {
    color: black !important;
    visibility: visible !important;
    display: inline-block !important;
    opacity: 1 !important;
    }

    </style>
    """, unsafe_allow_html=True)


    # Prediction button and logic
    if st.button("Kidney Test Result"):
        try:
            # Process input and make prediction
            final_input = []
            for key, val in user_inputs.items():
                val = val.strip().lower() if isinstance(val, str) else val
                if key in categorical_fields:
                    final_input.append(1 if val in ["yes", "present", "good", "normal"] else 0)
                else:
                    final_input.append(float(val))
            prediction = kidney_disease_model.predict([final_input])
            result = "⚠️ The person has kidney disease" if prediction[0] == 1 else "🟢 The person does not have kidney disease"
            st.success(result)
        except Exception as e:
            st.error(f"Input error: {e}")