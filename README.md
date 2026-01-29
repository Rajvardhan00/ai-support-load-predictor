# AI Support Load Predictor

An end-to-end Machine Learning application that predicts customer support load using historical and operational features.
The project is deployed as an interactive web application using Streamlit and Hugging Face Spaces, with the trained model managed separately via the Hugging Face Model Hub.

---

## Project Overview

Accurate forecasting of customer support load is critical for workforce planning, SLA compliance, and cost optimization.
This application allows users to input relevant features and obtain real-time predictions of expected support load using a trained machine learning model.

The project demonstrates:
- End-to-end ML workflow (training to deployment)
- Proper separation of code and model artifacts
- Production-style deployment practices

---

## Live Demo

Hugging Face Space:
https://huggingface.co/spaces/00Raj007/ai-support-load-predictor

---

## Architecture

├── streamlit_app.py        Streamlit application entry point  
├── requirements.txt        Python dependencies  
├── README.md               Project documentation  
├── models                  Local model directory (gitignored)  
└── .gitignore              Git ignore rules  

### Key Design Decisions
- No model binaries are committed to the repository
- Trained model is hosted on Hugging Face Model Hub
- Model is downloaded dynamically at runtime
- Ensures reproducibility, scalability, and clean Git history

---

## Model Management

The trained model (load_model.pkl) is stored in a dedicated Hugging Face Model repository:

https://huggingface.co/00Raj007/support-load-model

At runtime, the application downloads the model using huggingface_hub, caches it, and loads it for inference.

This follows standard MLOps and production deployment practices.

---

## Tech Stack

- Python 3.10
- Streamlit
- scikit-learn
- joblib
- Hugging Face Spaces
- Hugging Face Model Hub

---

## Local Setup

1. Clone the repository

git clone https://github.com/Rajvardhan00/support-load-predictor.git  
cd support-load-predictor  

2. Create a virtual environment (optional)

python -m venv venv  
source venv/bin/activate   (Linux / macOS)  
venv\Scripts\activate      (Windows)  

3. Install dependencies

pip install -r requirements.txt  

4. Run the application locally

streamlit run streamlit_app.py  

---

## Deployment (Hugging Face Spaces)

This project is deployed using Hugging Face Spaces with the Streamlit SDK.

Deployment steps:
1. Create a Hugging Face Space (Streamlit)
2. Add the required configuration block in README.md
3. Push the repository directly to the Space using Git
4. Do not include binary model files in the Space repository
5. Load the trained model dynamically from Hugging Face Model Hub

Deployment is triggered automatically on each git push.

---

## Important Notes

- Hugging Face Spaces do not allow binary model files in Git history
- Model files must be hosted externally (Model Hub)
- Deployment authentication uses Hugging Face access tokens
- GitHub is used only for source control, not for Space synchronization

---

## Future Improvements

- Add feature explanations and input validation
- Include confidence intervals for predictions
- Add monitoring for prediction drift
- Improve UI with charts and historical trends
- Add a reproducible training pipeline

---

## Author

Raj Vardhan  
Machine Learning and Software Engineering Enthusiast

---

