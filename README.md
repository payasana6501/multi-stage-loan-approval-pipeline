# Loan Approval Project -- 2 Stage ML Application

## Overview
This project implements a two-stage machine learning application for loan approval prediction. The stages are as follows:
1. **Classification Stage**: Classify applicants as either "Approved" or "Rejected" based on their application data.
2. **Regression Stage**: If the applicant is approved, predict the loan amount they are eligible for.

## Features
- **Data Preprocessing**: Handles missing values, categorical encoding, and feature scaling.
- **Modeling**: Utilizes `RandomForestClassifier` for classification and `GridSearchCV` for hyperparameter tuning.
- **Evaluation**: Provides detailed classification reports and accuracy metrics.

## Dataset
The dataset used for this project contains information about loan applicants, including their demographic details, financial history, and loan application status. The target variable for the classification stage is `loan_status`.

## Quick Start (Local)
Follow these steps to set up and run the project locally:

1. **Create a Virtual Environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**:
   ```bash
   python main.py
   ```

## Project Structure
- **analyze.ipynb**: Jupyter notebook for exploratory data analysis (EDA) and model experimentation.
- **main.py**: Main script to execute the loan approval pipeline.
- **requirements.txt**: Contains the list of dependencies required for the project.
- **loan_approval_dataset.csv**: Dataset used for training and testing the models.

## How It Works
1. The `main.py` script loads the dataset and preprocesses the data.
2. The classification model predicts whether a loan application is approved or rejected.
3. If approved, the regression model predicts the loan amount.
4. Outputs include classification metrics and predicted loan amounts.

## Dependencies
- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## Contributing
Contributions are welcome! Feel free to fork the repository and submit a pull request.

## License
This project is licensed under the MIT License.