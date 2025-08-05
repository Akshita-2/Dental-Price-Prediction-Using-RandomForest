# 🦷 Dental Price Predictor

An interactive web application that predicts the cost of common dental procedures using machine learning. Built with Streamlit and deployed on Streamlit Community Cloud, this tool helps users estimate consultation, scaling, filling, wisdom tooth extraction, and root canal costs based on real clinic data in Bangalore.

---

## 🔗 Live Demo

👉 [Click here to try the live demo](https://your-streamlit-link.streamlit.app)  
_(Replace with your actual Streamlit app link)_

---

## 📸 Screenshots

| Home Page | Prediction Results |
|-----------|-------------------|
| ![Home](images/home.png) | ![Results](images/results.png) |

---

## 🚀 Features

- 🔍 Predict prices for procedures like:
  - Consultation
  - Scaling
  - Filling
  - Wisdom Tooth Extraction
  - Root Canal Treatment (RCT)
- 📋 Interactive form based on clinic data
- 📊 Feature importance analysis (what impacts pricing most)
- 🎨 Custom animations (Lottie) and modern UI
- 🔒 Deployed on [Streamlit Community Cloud](https://streamlit.io/cloud)

---

## 🧠 Tech Stack

| Component       | Tech Used             |
|----------------|------------------------|
| Frontend       | Streamlit + Lottie     |
| ML Model       | Random Forest Regressor |
| Language       | Python                 |
| Data Handling  | Pandas, NumPy          |
| Deployment     | Streamlit Community Cloud |
| Visualization  | Vega-Lite Charts       |

---

## 🗂️ Dataset

- **Source**: Local Bangalore dental clinic data
- **Preprocessing**:
  - Cleaned missing values
  - Encoded categorical features
  - Converted binary responses to 1/0
- **Features**:
  - Clinic age, ownership, floors, chairs, staff count, amenities, reviews, accessibility, etc.
- **Targets**:
  - Prices for different dental procedures

---

## 📦 How to Run Locally

```bash
git clone https://github.com/yourusername/dental-price-predictor.git
cd dental-price-predictor
pip install -r requirements.txt
streamlit run app.py
