import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
from streamlit_lottie import st_lottie
import requests
import time
import joblib

# Set page config - MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Dental Clinic Price Predictor", 
    layout="wide", 
    page_icon="🦷",
    initial_sidebar_state="expanded"
)

# Cache Lottie animations with timeout
@st.cache_data(ttl=3600, show_spinner=False)
def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=3)
        if r.status_code == 200:
            return r.json()
        return None
    except:
        return None

# Load animations (cached)
lottie_tooth = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_eg3x8q.json")  # Smiling tooth
lottie_dental_chair = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_5tkzkblw.json")  # Dental chair
lottie_tooth_money = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_8wRE4I.json")  # Tooth with money
lottie_toothbrush = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_eg3x8q.json")  # Toothbrush
lottie_dentist = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_5tkzkblw.json")  # Dentist

# Custom CSS (cached)
@st.cache_data
def get_custom_css():
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4f1fe 100%);
    }
    .stApp {
        max-width: 1400px;
        margin: 0 auto;
        background: transparent;
    }
    .header {
        color: #2c3e50;
        padding: 1rem 0;
        text-align: center;
        background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    .subheader {
        color: #4b6cb7;
        text-align: center;
        font-weight: 400;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        margin-bottom: 25px;
        border-left: 5px solid #4b6cb7;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(0,0,0,0.15);
    }
    .metric-card h3 {
        color: #2c3e50;
        margin-top: 0;
        font-size: 1.1rem;
        font-weight: 600;
    }
    .metric-card h2 {
        color: #4b6cb7;
        margin: 10px 0;
        font-size: 2rem;
        font-weight: 700;
    }
    .metric-card p {
        color: #7f8c8d;
        margin-bottom: 0;
        font-size: 0.9rem;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #ffffff 0%, #f8f9fa 100%);
        box-shadow: 5px 0 15px rgba(0,0,0,0.05);
        border-right: none;
    }
    .prediction-card {
        background: white;
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        margin-bottom: 30px;
        transition: all 0.3s ease;
    }
    .prediction-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 30px rgba(0,0,0,0.15);
    }
    .stSelectbox, .stSlider, .stCheckbox {
        margin-bottom: 1rem;
    }
    .stButton>button {
        background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(75, 108, 183, 0.4);
    }
    .stDataFrame {
        border-radius: 10px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.05);
    }
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .fade-in {
        animation: fadeIn 0.8s ease forwards;
    }
    .pulse {
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    .highlight {
        background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    .tooltip {
        position: relative;
        display: inline-block;
        border-bottom: 1px dotted #4b6cb7;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #4b6cb7;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    </style>
    """

st.markdown(get_custom_css(), unsafe_allow_html=True)

# Cache data loading with optimized parameters
@st.cache_data(ttl=3600, show_spinner=False)
def load_data():
    with st.spinner('Optimizing data loading...'):
        df = pd.read_excel('dentalprice-100-pilot.xlsx')
        
        # Clean and preprocess data
        df['Clinic-Age'] = pd.to_numeric(df['Clinic-Age'], errors='coerce').fillna(df['Clinic-Age'].median())
        df['google review rating'] = pd.to_numeric(df['google review rating'], errors='coerce').fillna(df['google review rating'].median())
        
        # Convert yes/no to binary
        binary_cols = ['emergency attending (yes/no)', 'drinking water available(yes/no)', 
                       'Is pharma with in(y/n)', 'are dental accessories sold(y/n)', 
                       'LED display board of clinic(y/n)', 'air conditioned(y/n)', 
                       'website(y/n)', 'FB page(y/n)', 'Linked page (y/n)', 
                       'youtube channel(y/n)', 'sms mktg', 'FM radio mktg(y/n)', 
                       'mostly visit by appointment (y/n)']
        
        for col in binary_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.lower().map({'yes': 1, 'no': 0, 'y': 1, 'n': 0}).fillna(0)
        
        # Convert categorical variables
        if 'owned/ rental/leased' in df.columns:
            df['owned/ rental/leased'] = df['owned/ rental/leased'].astype(str).str.lower().map({
                'owned': 0, 'rental': 1, 'leased': 2}).fillna(0)
        
        if 'accessability from main road(good/average/poor)' in df.columns:
            df['accessability from main road(good/average/poor)'] = df['accessability from main road(good/average/poor)'].astype(str).str.lower().map({
                'good': 0, 'average': 1, 'poor': 2}).fillna(0)
        
        # Parking availability
        if 'parking(no parking/no. of two wheeler/no. of four wheeler)' in df.columns:
            df['parking'] = df['parking(no parking/no. of two wheeler/no. of four wheeler)'].str.contains('available', case=False).astype(int)
        
        # Fill missing numeric values with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
        
        return df

# Cache address mapping
@st.cache_data
def create_address_mapping(df):
    address_map = {}
    defaults = {
        'Clinic-Age': df['Clinic-Age'].median(),
        'owned/ rental/leased': 0,
        'no. of floors occupied': 1,
        'no. of dental chairs': 2,
        'no.of dental x ray machines': 1,
        'no. of LCD/TV s in clinic': 1,
        'no.of fans': 2,
        'no. of female doctors': 2,
        'no. of female staff': 2,
        'google review rating': df['google review rating'].median(),
        'parking': 1,
        'accessability from main road(good/average/poor)': 0,
        'emergency attending (yes/no)': 1,
        'air conditioned(y/n)': 1,
        'mostly visit by appointment (y/n)': 1,
        'clinic name': 'Unknown Clinic'
    }
    
    for _, row in df.iterrows():
        address = row.get('address', 'Unknown Address')
        clinic_data = defaults.copy()
        for key in defaults:
            if key in row:
                clinic_data[key] = row[key]
        address_map[address] = clinic_data
    
    return address_map

# Cache model training with optimized parameters
@st.cache_resource(show_spinner=False)
def train_models(df):
    # Features to use for prediction
    features = [
        'Clinic-Age', 'owned/ rental/leased', 'no. of floors occupied', 
        'no. of dental chairs', 'no.of dental x ray machines', 
        'no. of LCD/TV s in clinic', 'no.of fans', 'no. of female doctors', 
        'no. of female staff', 'google review rating', 'parking',
        'accessability from main road(good/average/poor)', 
        'emergency attending (yes/no)', 'air conditioned(y/n)',
        'mostly visit by appointment (y/n)'
    ]
    
    # Ensure features exist in dataframe
    features = [f for f in features if f in df.columns]
    
    # Target variables
    targets = {
        'consultancy charges': 'Consultation',
        'Scaling charges': 'Scaling',
        'Filling charges': 'Filling',
        'Wisdom tooth extraction': 'Wisdom Tooth Extraction',
        'RCT (without cap) charges': 'Root Canal Treatment'
    }
    
    models = {}
    
    for target_col, target_name in targets.items():
        if target_col not in df.columns:
            continue
            
        # Prepare data
        X = df[features]
        y = df[target_col]
        
        # Remove rows with missing target values
        valid_idx = y.notna()
        X = X[valid_idx]
        y = y[valid_idx]
        
        if len(X) == 0:
            continue
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        try:
            # Train model with optimized parameters
            model = RandomForestRegressor(
                n_estimators=50,  # Reduced for faster training
                max_depth=10,
                random_state=42,
                n_jobs=-1  # Use all CPU cores
            )
            model.fit(X_train, y_train)
            
            models[target_name] = {
                'model': model,
                'features': features,
                'target_col': target_col
            }
        except Exception as e:
            st.warning(f"Could not train model for {target_name}: {str(e)}")
            continue
    
    return models

# Load data and models (cached)
df = load_data()
address_map = create_address_mapping(df)
models = train_models(df)

# Header with animation
col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.markdown("<h1 class='header'>🦷 Dental Clinic Price Predictor</h1>", unsafe_allow_html=True)
    st.markdown("""
    <p class='subheader'>
    Predict prices for common dental procedures based on clinic characteristics in Bangalore
    </p>
    """, unsafe_allow_html=True)
    if lottie_tooth:
        st_lottie(lottie_tooth, height=150, key="tooth")

# Sidebar with clinic information
with st.sidebar:
    st.markdown("<h2 style='color: #4b6cb7; text-align: center;'>Clinic Information</h2>", unsafe_allow_html=True)
    
    if lottie_dental_chair:
        st_lottie(lottie_dental_chair, height=150, key="dental_chair")
    
    # Location section
    st.markdown("### 📍 Location Details")
    address_options = list(address_map.keys())
    selected_address = st.selectbox("Select Clinic Address", address_options)
    
    # Get default values from selected address
    clinic_data = address_map.get(selected_address, {})
    
    # Basic info section
    st.markdown("### ℹ️ Basic Information")
    col1, col2 = st.columns(2)
    with col1:
        clinic_age = st.slider('Clinic Age (years)', 1, 50, int(clinic_data.get('Clinic-Age', 10)), help="How many years has the clinic been operating?")
    with col2:
        ownership = st.selectbox('Ownership', ['Owned', 'Rental', 'Leased'], 
                               index=int(clinic_data.get('owned/ rental/leased', 1)), help="Ownership status of the clinic premises")
    
    # Facility info section
    st.markdown("### 🏥 Facility Information")
    col1, col2 = st.columns(2)
    with col1:
        floors = st.slider('Floors occupied', 1, 3, int(clinic_data.get('no. of floors occupied', 1)), help="Number of floors the clinic occupies")
        chairs = st.slider('Dental chairs', 1, 5, int(clinic_data.get('no. of dental chairs', 2)), help="Number of dental chairs available")
        xray = st.slider('X-ray machines', 0, 2, int(clinic_data.get('no.of dental x ray machines', 1)), help="Number of dental X-ray machines")
    with col2:
        tvs = st.slider('LCD/TVs in clinic', 0, 10, int(clinic_data.get('no. of LCD/TV s in clinic', 1)), help="Number of LCD/TV screens in the clinic")
        fans = st.slider('Number of fans', 0, 10, int(clinic_data.get('no.of fans', 2)), help="Number of fans in the clinic")
    
    # Staff info section with toothbrush animation
    st.markdown("### 👩‍⚕️ Staff Information")
    col1, col2, col3 = st.columns([1,1,0.2])
    with col1:
        female_docs = st.slider('Female doctors', 0, 10, int(clinic_data.get('no. of female doctors', 2)), help="Number of female dentists")
    with col2:
        female_staff = st.slider('Female staff', 0, 10, int(clinic_data.get('no. of female staff', 2)), help="Number of female support staff")
    with col3:
        if lottie_toothbrush:
            st_lottie(lottie_toothbrush, height=60, key="toothbrush")
    
    # Ratings and amenities
    st.markdown("### ⭐ Ratings & Amenities")
    rating = st.slider('Google rating (1-5)', 1.0, 5.0, float(clinic_data.get('google review rating', 4.5)), 0.1, help="Clinic's Google review rating")
    access = st.selectbox('Road accessibility', ['Good', 'Average', 'Poor'], 
                         index=int(clinic_data.get('accessability from main road(good/average/poor)', 0)), help="How accessible is the clinic from main road?")
    
    col1, col2 = st.columns(2)
    with col1:
        parking = st.checkbox('🚗 Parking available', value=bool(clinic_data.get('parking', True)))
        emergency = st.checkbox('🚨 Emergency services', value=bool(clinic_data.get('emergency attending (yes/no)', True)))
    with col2:
        ac = st.checkbox('❄️ Air conditioned', value=bool(clinic_data.get('air conditioned(y/n)', True)))
        appointment = st.checkbox('📅 Appointment preferred', value=bool(clinic_data.get('mostly visit by appointment (y/n)', True)))

# Map inputs to model features
ownership_map = {'Owned': 0, 'Rental': 1, 'Leased': 2}
access_map = {'Good': 0, 'Average': 1, 'Poor': 2}

input_data = {
    'Clinic-Age': clinic_age,
    'owned/ rental/leased': ownership_map[ownership],
    'no. of floors occupied': floors,
    'no. of dental chairs': chairs,
    'no.of dental x ray machines': xray,
    'no. of LCD/TV s in clinic': tvs,
    'no.of fans': fans,
    'no. of female doctors': female_docs,
    'no. of female staff': female_staff,
    'google review rating': rating,
    'parking': 1 if parking else 0,
    'accessability from main road(good/average/poor)': access_map[access],
    'emergency attending (yes/no)': 1 if emergency else 0,
    'air conditioned(y/n)': 1 if ac else 0,
    'mostly visit by appointment (y/n)': 1 if appointment else 0
}

# Convert to DataFrame for prediction
input_df = pd.DataFrame([input_data])

# Make predictions
with st.container():
    st.markdown("<h2 style='color: #4b6cb7; text-align: center;'>💰 Predicted Prices</h2>", unsafe_allow_html=True)
    clinic_name = address_map.get(selected_address, {}).get('clinic name', 'Selected Clinic')
    st.markdown(f"<p style='color: #555; text-align: center;'>For <strong class='highlight'>{clinic_name}</strong> at: <strong>{selected_address}</strong></p>", unsafe_allow_html=True)
    
    if lottie_tooth_money:
        st_lottie(lottie_tooth_money, height=100, key="money")
    
    if not models:
        st.error("No models were successfully trained. Please check your data.")
    else:
        predictions = {}
        for proc, model_info in models.items():
            try:
                model = model_info['model']
                pred = model.predict(input_df)[0]
                accuracy_range = round(pred * 0.15)  # Calculate 15% of predicted price
                predictions[proc] = {
                    'price': round(pred),
                    'accuracy': accuracy_range
                }
            except Exception as e:
                st.warning(f"Could not make prediction for {proc}: {str(e)}")
                continue
        
        # Display predictions in cards
        cols = st.columns(len(predictions))
        for idx, (proc, pred_info) in enumerate(predictions.items()):
            with cols[idx]:
                st.markdown(f"""
                <div class="metric-card fade-in" style="animation-delay: {idx*0.2}s">
                    <h3>{proc}</h3>
                    <h2>₹{pred_info['price']:,}</h2>
                    <p style="color: #666;">±₹{pred_info['accuracy']:,} (±15%)</p>
                </div>
                """, unsafe_allow_html=True)

# Data insights section
with st.container():
    st.markdown("<h2 style='color: #4b6cb7; text-align: center; margin-top: 40px;'>📊 Market Insights</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h3 style='color: #4b6cb7;'>📈 Average Prices</h3>", unsafe_allow_html=True)
        try:
            avg_prices = df[['consultancy charges', 'Scaling charges', 'Filling charges', 
                           'Wisdom tooth extraction', 'RCT (without cap) charges']].mean().round()
            
            avg_prices = avg_prices.rename({
                'consultancy charges': 'Consultation',
                'Scaling charges': 'Scaling',
                'Filling charges': 'Filling',
                'Wisdom tooth extraction': 'Wisdom Tooth Extraction',
                'RCT (without cap) charges': 'Root Canal Treatment'
            })
            
            # Style the dataframe
            styled_avg = avg_prices.rename('Average Price (₹)').reset_index().rename(columns={'index': 'Procedure'})
            styled_avg = styled_avg.style \
                .background_gradient(cmap='Blues') \
                .set_properties(**{'border': '1px solid #4b6cb7', 'border-radius': '5px'}) \
                .format({'Average Price (₹)': '₹{:,.0f}'})
            
            st.dataframe(styled_avg, use_container_width=True)
        except Exception as e:
            st.error(f"Could not calculate average prices: {str(e)}")
    
    with col2:
        st.markdown("<h3 style='color: #4b6cb7;'>🏆 Price Influencers</h3>", unsafe_allow_html=True)
        procedure = st.selectbox('Select procedure to analyze', list(models.keys()), key='procedure_select')
        
        try:
            model = models[procedure]['model']
            features = models[procedure]['features']
            importances = model.feature_importances_
            importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
            importance_df = importance_df.sort_values('Importance', ascending=False).head(5)
            
            # Create a styled bar chart
            st.vega_lite_chart(importance_df, {
                'mark': {'type': 'bar', 'cornerRadiusEnd': 4, 'color': '#4b6cb7'},
                'encoding': {
                    'x': {'field': 'Importance', 'type': 'quantitative', 'title': 'Importance'},
                    'y': {
                        'field': 'Feature', 
                        'type': 'nominal', 
                        'title': 'Feature',
                        'sort': '-x',
                        'axis': {'labelLimit': 100}
                    },
                },
                'height': 300,
            }, use_container_width=True)
            
            st.caption(f"Top factors affecting {procedure.lower()} pricing")
        except Exception as e:
            st.error(f"Could not show feature importance: {str(e)}")

# Raw data option
with st.expander("🔍 Show raw data and statistics", expanded=False):
    st.markdown("<h3 style='color: #4b6cb7;'>📂 Dataset Overview</h3>", unsafe_allow_html=True)
    st.write(f"Using data from {len(df)} dental clinics in Bangalore")
    
    tab1, tab2 = st.tabs(["📋 Raw Data", "📊 Statistics"])
    
    with tab1:
        if st.checkbox('Show raw data', key='raw_data'):
            st.dataframe(df.style.background_gradient(cmap='Blues'), use_container_width=True)
    
    with tab2:
        if st.checkbox('Show statistics', key='stats'):
            st.write(df.describe().style.background_gradient(cmap='Blues'))

# Footer with dentist animation
st.markdown("---")
col1, col2, col3 = st.columns([1,2,1])
with col2:
    if lottie_dentist:
        st_lottie(lottie_dentist, height=150, key="dentist")
st.markdown("""
<div style="text-align: center; color: #7f8c8d; font-size: 0.9rem; margin-top: 20px;">
    <p>🦷 Dental Clinic Price Predictor | Powered by Machine Learning</p>
    <p>For accurate pricing information, please consult with dental clinics directly</p>
</div>
""", unsafe_allow_html=True)