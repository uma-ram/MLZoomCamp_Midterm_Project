# Medical Cost Prediction
This a mid term project repo 
##Conda Env
conda create -n medical_cost_env python=3.12
conda activate medical_cost_env

# Medical Cost Prediction

## Project Overview

This project aims to build a machine learning model that predicts individual medical insurance costs based on personal attributes such as age, sex, BMI, number of children, smoking status, and region. Using the publicly available Medical Cost Personal Dataset from Kaggle, the project involves:

- **Data Exploration and Preprocessing:** Understanding the dataset, handling categorical variables, and preparing features for modeling.  
- **Model Development:** Training regression models (Linear Regression and XGBoost) to predict medical charges accurately.  
- **API Development:** Creating a RESTful API using FastAPI to serve the trained model and provide cost predictions based on user input.  
- **Containerization:** Packaging the application into a Docker container for easy deployment and scalability.  
- **Future Extensions:** Adding a user-friendly interface using Streamlit to allow non-technical users to interact with the model.

This project demonstrates the end-to-end workflow of a machine learning application, from data handling and model training to deployment and serving predictions via an API.
