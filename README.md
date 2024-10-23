# Patient Insurance Cost Predictor

This project uses machine learning to predict medical insurance costs based on various factors like age, BMI, smoking status, and other features. The project involves exploratory data analysis, building predictive models using Linear Regression and Random Forest, and evaluating the models’ performance.

## Project Overview

1. **Data Analysis**: 
   - Exploratory data analysis is performed to understand the distribution of different features and their relationships with medical costs.
   
2. **Modeling**: 
   - A **Linear Regression** model is built to predict insurance costs.
   - A **Random Forest Regressor** model is used as an alternative method to improve the predictions.
   
3. **Evaluation**: 
   - Models are evaluated using metrics like **Mean Squared Error (MSE)** and **R² Score** to compare their performances.
   
4. **Visualization**: 
   - Several visualizations are used to illustrate the data, feature importance, and model predictions.

## Files

- `Patient_insurance_cost_predictor_.ipynb`: The Jupyter Notebook containing the code for data analysis, modeling, and visualizations.
- `insurance.csv`: The dataset containing patient information such as age, BMI, smoking status, and insurance charges.
- `patient_cost_predictor.pkl`: The saved Random Forest model used to predict insurance costs.

## Analysis Pipeline

The analysis workflow is as follows:

1. **Data Loading and Preprocessing**:
   - The dataset is loaded and basic data analysis (head, info, describe) is performed to understand its structure.
   - Missing values are checked and handled.
   - Categorical variables (such as sex, smoker, and region) are converted to numerical values.

2. **Exploratory Data Analysis**:
   - Distribution of features like **age** and **charges** is visualized using histograms.
   - Pair plots are generated to explore relationships between features.

3. **Model Building**:
   - The data is split into training and testing sets using **train_test_split**.
   - **Linear Regression** is used to build a baseline model to predict insurance charges.
   - **Random Forest Regressor** is used to build a more robust model for predictions.

4. **Evaluation**:
   - The performance of the models is evaluated using:
     - **Mean Squared Error (MSE)**
     - **R² Score**
   - Predictions are visualized by plotting actual charges vs. predicted charges.

5. **Feature Importance**:
   - The importance of features is calculated using the Random Forest model and visualized through a bar plot.

6. **Model Saving**:
   - The Random Forest model is saved as a `.pkl` file using `joblib` for future use.

## Running the Script

To run the notebook, ensure that you have the necessary libraries installed. You can install the required dependencies by running the following command:

```bash
pip install -r requirements.txt
