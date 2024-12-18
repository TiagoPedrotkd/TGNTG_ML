# Machine Learning Project 2024/2025: Deciding on Compensation Benefits

This repository contains the code and documentation for the Machine Learning course project, aimed at creating a multiclass classification model to accurately predict the type of decision made by the New York Workers' Compensation Board (WCB) in workplace injury claims.

## Project Objectives
- **Multiclass Classification**: Develop and evaluate machine learning models to predict the type of injury assigned to claims based on data provided by the WCB.
- **Model Optimization**: Explore strategies to enhance model performance, including hyperparameter tuning and feature selection techniques.
- **Additional Insights**: Investigate and implement analyses beyond the primary classification goal, such as:
  - Feature importance for decision-making.
  - Creating a prediction interface for new inputs.
  - Experimenting with auxiliary variables to improve model performance.

## Repository Structure
- **Code Notebook**: All the code used in the project, including preprocessing, training, optimization, and analysis of models.
- **Final Report**: A detailed summary of the project, explaining decisions made and results achieved.
- **Datasets**: Information on the training and testing data used (provided by the WCB, subject to privacy guidelines).
- **Other Resources**: Auxiliary scripts, visualizations, and additional documentation.

## Technologies Used
- Python and popular libraries like Pandas, Scikit-learn, and Matplotlib.
- Cross-validation and standardized metrics for model evaluation.
- Kaggle competition for benchmarking results.

## About the Data
The data covers workplace injury claims from 2020 to 2022, providing attributes such as age, injury type, location, and related medical information. The main goal is to predict the "Claim Injury Type," a multiclass variable.

## Contributions
This project was developed as part of the Machine Learning 2024/2025 course at Nova IMS. We thank the New York Workers' Compensation Board for providing the data and supporting this project.

## License
This project is licensed under the [MIT License](LICENSE). Feel free to use, modify, and distribute it as long as the original license is maintained.

## How to Run
1. Clone the repository;
2. Navigate to the project directory;
3. Install python 3.11.0;
4. Create a enviorement:
  ```bash
   python3.11 -m venv venv
5. Activate enviorement:
  ```bash
  .\venv\Script\activate
6. Install dependencies:
   ```bash
   pip install -r requirements.txt
7. Open the Jupyter Notebook;
8. Run the notebook to train and test the models;
 

