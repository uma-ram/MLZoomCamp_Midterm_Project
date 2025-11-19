# Medical Insurance Cost Prediction


## Problem Description

Healthcare costs are a significant concern for individuals and insurance providers. Accurately estimating medical insurance charges based on personal attributes helps in budgeting, planning, and risk assessment.

The goal of this project is to develop a regression model that predicts the medical insurance cost for an individual using features such as:

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
- **Model Development:** Training regression models (Linear Regression, Random Forest, Gradien Boosting and XGBoost) to predict medical charges accurately.  
- **API Development:** Creating a RESTful API using FastAPI to serve the trained model and provide cost predictions based on user input.  
- **Validation:** Implementing validation for feature during prediction using Pydantic.
- **Containerization:** Packaging the application into a Docker container for easy deployment and scalability.  
- **Future Extensions:** Adding a user-friendly interface using Streamlit to allow non-technical users to interact with the model.

This project demonstrates the end-to-end workflow of a machine learning application, from data handling and model training to deployment and serving predictions via an API.

## Project Structure
>
📁 MLZoomcamp_Midterm_Project<br>
├── data/<br>
│   └──insurance.csv          # Training dataset<br>
├── Sceenshots/<br>
│   └── API up & running.png <br>
│   └── Docker server up & running.png<br>
│   └── Model Creation.png<br>
│   └── Pydantic Validation Check.png<br>
│   └── Successfull Model Prediction.png<br>
│── .python-version <br>
│── Dockerfile<br>
│── README.md <br>
│── custom_transformer.py <br>
│── Model.bin<br>
│── notebook.ipynb <br>
│── predict.py <br>
│── pyproject.toml <br>
│── requirements.txt <br>
│── train.py <br>
│── uv.lock<br>


## Exploratory Data Analysis

### Preprocessing
   * Categorical Features: Handled using DictVectorizer for one-hot encoding.
   * Numerical Features: Scaled using StandardScaler to ensure they have a zero mean and unit variance.

## Training Models

GRadient Boosting Regressor is selected as best performing model after comparing with other different models and trained.

<table>
  <tr style="background-color:#f2f2f2; font-weight:bold;">
    <td>Model</td>
    <td>RMSE</td>
    <td>MAE</td>
    <td>R²</td>
  </tr>

  <tr style="background-color:white;">
    <td>Gradient Boosting</td>
    <td>5031.21</td>
    <td>2732.33</td>
    <td>0.82</td>
  </tr>

  <tr style="background-color:#e7f0fd;">
    <td>Random Forest</td>
    <td>5210.64</td>
    <td>2879.37</td>
    <td>0.81</td>
  </tr>

  <tr style="background-color:white;">
    <td>XGBoost</td>
    <td>5760.64</td>
    <td>3235.07</td>
    <td>0.77</td>
  </tr>

  <tr style="background-color:#e7f0fd;">
    <td>Linear Regression</td>
    <td>6196.21</td>
    <td>4338.69</td>
    <td>0.73</td>
  </tr>
</table>


## Hyperparameter Tuning

<p> First 2 models(Gradient Boosting, Random Forest) are seleted for hyperparamater tuning. Gradient Boosting Regressor is selected as best model for training and prediction.Th result is shown below </p>

<table>
  <tr style="background-color:#f2f2f2; font-weight:bold;">
    <td>Model</td>
    <td>RMSE</td>
    <td>MAE</td>
    <td>R²</td>
  </tr>
  
  <tr style="background-color:white;">
    <td>Gradient Boosting</td>
    <td>4891.91</td>
    <td>2781.05</td>
    <td>0.83</td>
  </tr>

  <tr style="background-color:#e7f0fd;">
    <td><b>Random Forest</b></td>
    <td><b>4997.33</b></td>
    <td><b>2729.19</b></td>
    <td><b>0.83</b></td>
  </tr>
</table>


## How to Run Locally

### Prerequisites ###
   * `uv` package manager installed.
   * `Docker` installed (for Docker deployment).

### Local Setup
   * Clone the repository and navigate to the project directory. <br>

   * Set up the environment: <br>
      > uv sync
   
   * Train the model (already model is trained and available in repository)
      > uv run python train.py

   * Run the FastAPI server
      > uv run uvicorn predict:app --reload --host 0.0.0.0 --port 8000 


## Docker
 
 * Build Image
    > docker build -t insurance-prediction .
 
 * Run Image
    > docker run -it --rm -p 8000:8000 insurance-prediction

 * Visit locally<br>
    👉 [http://localhost:8000](http://localhost:8000)  <br>
    👉 [http://localhost:8000/docs](http://localhost:8000/docs)

## Screen Shots

   * <B>Model created and saved successfully</B><br>

      ![API Server Running](</Screenshots/Model Creation.png>)  

   * <b>Docker Server Up &  Running </b> <br>  

      ![API Server Running](</Screenshots/API UP & Running.png>)  

   * <b>Successful Model Prediction<b> <br>

      ![API Server Running](</Screenshots/Successfull Model Prediction.png>)  

   * <b>Docker Server Up & Running</b> <br>

      ![API Server Running](</Screenshots/Docker server up & running.png>)  

   * <b>Pydantic Validation</b> <br>

      ![API Server Running](</Screenshots/Pydantic Validation Check.png>)  

## API Call 

   You can make a POST request to the /predict endpoint to get a prediction.

Using curl: <br>

   ```
   curl -X 'POST' \
   'http://127.0.0.1:8000/predict' \
   -H 'accept: application/json' \
   -H 'Content-Type: application/json' \
   -d '{
   "age": 38,
   "sex": "male",
   "bmi": 15,
   "children": 2,
   "smoker": "no",
   "region": "southwest"
   }'
   ```
<b> Response
   ```
   {
      "predicted_medical_cost": 7601.16
   }
   ```

## Cloud Deployment

   Cloud Deployment is done. Deployed in Render.com and service was running successfully  

   * <b>Successful Model Deployment on cloud (render.com)<b> <br>

      ![Cloud Server Running](</Screenshots/Cloud Deployment.png>)<br>

   * <b>Service started and predicted insurance cost for the given user.<b> <br>

      ![Prediction](</Screenshots/Cloud Service Started.png>)<br>

      ![Prediction](</Screenshots/Successfull Model Prediction.png>)



