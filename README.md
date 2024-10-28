# Weather Classification using Machine Learning

This project applies machine learning algorithms to classify weather types based on meteorological data. Using various data preprocessing, visualization, and modeling techniques, this project aims to predict weather categories like Rainy, Sunny, Cloudy, and Snowy with high accuracy.

## Project Overview
Weather classification has applications in agriculture, transportation, and disaster management. This project explores different machine learning models to identify weather types based on features like temperature, humidity, wind speed, atmospheric pressure, and more. 

## Dataset
The dataset contains weather-related attributes:
- **Features**: Temperature, Humidity, Wind Speed, Precipitation (%), Cloud Cover, Atmospheric Pressure, UV Index, Season, Visibility, Location
- **Target**: Weather Type (e.g., Sunny, Rainy, Cloudy, Snowy)

## Data Preprocessing
- Removed duplicates and handled missing values.
- Encoded categorical features and removed outliers.
- Performed feature selection and data normalization.

## Modeling
Implemented the following classification models:
- **Logistic Regression**: Achieved accuracy after tuning for class balance.
- **Decision Tree**: Prone to overfitting without pruning.
- **Random Forest**: Improved accuracy by increasing estimators.
- **SVM** and **XGBoost**: Showed high generalization with proper tuning.

## Results
The models' accuracies varied based on tuning:
- Logistic Regression: 89% 
- Decision Tree: 89.63%
- Random Forest: 90%
- SVM: 89.24%
- XGBoost: 89.28%


## Getting Started
To replicate the analysis:
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run the Jupyter notebook in `notebooks/`.

## References
- [Kaggle](https://www.kaggle.com/code/dogukantabak/weather-type-classification) - Dataset source.
