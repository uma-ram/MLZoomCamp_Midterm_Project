# Medical Insurance Cost Prediction


## Problem Description

Healthcare costs are a significant concern for individuals and insurance providers. Accurately estimating medical insurance charges based on personal attributes helps in budgeting, planning, and risk assessment.

The goal of this project is to develop a machine learning model that predicts the medical insurance cost for an individual using features such as:

- Age  
- Sex  
- Body Mass Index (BMI)  
- Number of children  
- Smoking status  
- Residential region  

By analyzing historical insurance data, the model will learn patterns and relationships between these features and the insurance charges, enabling accurate cost predictions for new individuals.

This predictive tool can assist insurance companies in pricing policies fairly and individuals in understanding their potential healthcare expenses.
________________________________________________________________________________

## Project Overview

This project aims to build a machine learning model that predicts individual medical insurance costs based on personal attributes such as age, sex, BMI, number of children, smoking status, and region. Using the publicly available Medical Cost Personal Dataset from Kaggle, the project involves:

- **Data Exploration and Preprocessing:** Understanding the dataset, handling categorical variables, and preparing features for modeling.  
- **Model Development:** Training regression models (Linear Regression, Random Forest and XGBoost) to predict medical charges accurately.  
- **API Development:** Creating a RESTful API using FastAPI to serve the trained model and provide cost predictions based on user input.  
- **Containerization:** Packaging the application into a Docker container for easy deployment and scalability.  
- **Future Extensions:** Adding a user-friendly interface using Streamlit to allow non-technical users to interact with the model.

This project demonstrates the end-to-end workflow of a machine learning application, from data handling and model training to deployment and serving predictions via an API.


To run the train.py
python train.py

3. Build image again
docker build -t insurance-prediction .

4. Run container
docker run -it --rm -p 8000:8000 insurance-prediction


Then visit:

👉 http://localhost:8000

👉 http://localhost:8000/docs

