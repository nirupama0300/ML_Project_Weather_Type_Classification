Machine Learning Project - Weather Classification using Machine Learning

Introduction:
Weather classification is essential for various applications, including agriculture, transportation, and disaster management. This project aims to predict different weather types based on meteorological data using various machine learning algorithms. The dataset used contains features such as temperature, humidity, wind speed, and atmospheric pressure.
By exploring and preprocessing the data, and applying different classification algorithms, we aim to achieve high predictive accuracy and understand the relationships between different weather parameters and the impact of features on the weather.

Dataset Description:
The dataset contains synthetic data for weather type classification, including features like temperature, humidity, wind speed, precipitation percentage, cloud cover, atmospheric pressure, UV index, season, visibility, and location. Ideal for practising classification algorithms and exploring weather pattern predictions.
It includes various weather-related features and categorises the weather into four types: Rainy, Sunny, Cloudy, and Snowy. This dataset is designed for practising classification algorithms, data preprocessing, and outlier detection methods.
The dataset includes several columns:
Temperature: The current temperature (°C).
Humidity: The percentage of humidity.
Wind Speed: Speed of wind (km/h).
Precipitation (%): Percentage of precipitation.
Atmospheric Pressure: Pressure in hPa.
UV Index: The level of ultraviolet radiation.
Visibility (km): Visibility in kilometres.
Season: The season during the observation (Spring, Summer, Autumn, Winter).
Location: The geographical location of the observation.
Cloud Cover: The percentage of cloud cover.
Weather Type: The target variable representing different weather types (e.g., Sunny, Rainy, Snowy).
Data Preprocessing:
Data Loading:
The dataset was loaded using pandas, and initial inspections revealed useful statistics and insights, including:
Data Information: Data types and null values. Used data.info()
Descriptive Statistics: Basic statistics of numerical features. Used data.describe()
Duplicate Records: Checked for and handled any duplicate entries. Used data.duplicated().sum() and found that there were duplicates present. They were eventually removed.
Data Visualisation
Visualisations were created using seaborn and matplotlib to understand the distribution of features:
Count plots for categorical variables like Season, Cloud Cover, Location, and Weather Type.
Line plots to analyse relationships between numerical features, such as Temperature vs. Humidity and Wind Speed.
UNIVARIATE ANALYSIS:
Count Plots
Used seaborn's countplot to visualise the distribution of categorical variables:
Cloud Cover: Understanding the frequency of different cloud cover levels.
Season: Observing the number of observations per season.
Location: Checking the distribution of data across different locations.
Weather Type: Analysing the frequency of each weather type in the dataset.
These plots helped identify any class imbalances that might affect model training.
Histograms
Plotted histograms for numerical variables using data.hist() to observe their distributions and identify any skewness or kurtosis that might require transformation.
BIVARIATE ANALYSIS
Line Plots
Examined relationships between temperature and other continuous variables:
Temperature vs. Humidity: To see how temperature variations affect humidity levels.
Temperature vs. Wind Speed: Understanding the correlation between temperature and wind speed.
Temperature vs. Precipitation (%): Investigating how temperature impacts precipitation probability.
Temperature vs. Atmospheric Pressure: Analysing the interplay between temperature and atmospheric pressure.
Temperature vs. UV Index: Observing how temperature relates to UV exposure.
Visibility vs. Wind Speed: Exploring the effect of wind speed on visibility.
These plots provided insights into potential predictors for the weather type.
Analysis
Analysed how different weather parameters change with seasons by plotting:
Season vs. Humidity
Season vs. Wind Speed
Season vs. Precipitation (%)
Season vs. Atmospheric Pressure
Season vs. UV Index
Season vs. Temperature
This helped in understanding seasonal patterns that could influence weather classification.
Other Visualisations
Joint Plot of Wind Speed vs. Temperature: Using sns.jointplot to examine the relationship and density.
Regression Plot of Atmospheric Pressure vs. Temperature: To identify linear relationships.
Box Plots: Comparing distributions of wind speed across seasons and humidity across weather types to detect outliers and variations.
Correlation Matrix
Computed and visualised the correlation matrix using sns.heatmap. This helped in identifying the strength and direction of relationships between features, which is crucial for detecting multicollinearity that could affect model performance. From the correlation matrix:
Temperature showed a moderate positive correlation with UV Index (0.37) and a weak positive correlation with Visibility (0.25). However, its correlation with Precipitation was negative (-0.29), indicating that lower temperatures are generally associated with higher precipitation levels.
Precipitation had a strong negative correlation with Visibility (-0.46), suggesting that as precipitation increases, visibility decreases. This is intuitive, as heavy rain or snow often reduces visibility.
UV Index had a moderate positive correlation with Visibility (0.36), which could indicate clearer weather conditions are associated with higher UV exposure. It also showed a weak positive relationship with Weather_Type1 (0.35).
Weather_Type1, representing weather categories, exhibited weak correlations with all the other features. This suggests that Weather_Type1 is not heavily influenced by any single variable but may still contribute valuable insights for classification models.
Identifying these correlations allows us to decide whether to handle multicollinearity (e.g., by removing or combining features), especially when building models like Logistic Regression that assume little to no multicollinearity among independent variables.
Data Preprocessing:
Label Encoding
Categorical variables were encoded using LabelEncoder:
Cloud Cover
Season
Location
Weather Type
This conversion was necessary for algorithms that require numerical input.
Feature Selection
I dropped less relevant features based on the correlation matrix and domain knowledge, such as, cloud cover, season, weather type, location, humidity, wind speed and atmospheric pressure.
The final DataFrame df included the most impactful features for modelling.
Handling Outliers
Outlier Detection
Defined a function det_outliers to detect outliers in the numerical features using the Interquartile Range (IQR) method. Outliers were printed for each feature.
Outlier Visualisation
Plotted the outliers using a custom function plot_outliers. Scatter plots with horizontal lines indicating the lower and upper bounds were generated for each feature.
Outlier Removal
Removed outliers using the Z-score method from the scipy library:
Calculated Z-scores for each observation.
Filtered out observations where the absolute Z-score was greater than 3.
This resulted in a cleaner dataset (cleaned_df) with reduced noise.
Modelling:
Data Splitting
I have split the data into training and testing sets using train_test_split:
Features (X): All columns except Weather_Type1.
Target (y): The encoded Weather_Type1 column.
Test Size: 20%
Random State: 42 (for reproducibility)
Logistic Regression
Initial Model
Trained a logistic regression model using LogisticRegression with default parameters. However, the following was  observed:
Low Training Accuracy: Indicating underfitting.
Hyperparameter Tuning
To improve the model,   hyperparameter tuning using GridSearchCV was performed:

Parameters Tuned:
Regularisation strength (C)
Solver (liblinear, saga)
Pipeline: Included StandardScaler for feature scaling.
Class Weight: Set to 'balanced' to handle any class imbalances.
The best model from grid search showed:
Improved Training Accuracy: Indicating better generalisation.
Consistent Test Accuracy: Confirming model reliability.
Cross-Validation
Evaluated the model using 5-fold cross-validation to ensure robustness. The average cross-validation score was satisfactory.
Evaluation Metrics
Confusion Matrix: Showed correct and incorrect classifications.
Classification Report: Provided precision, recall, and F1-score for each class.
Decision Tree Classifier
Initial Model
A decision tree classifier was trained using DecisionTreeClassifier without parameter tuning, resulting in:
High Training Accuracy: Indicating overfitting.
Low Test Accuracy: Due to the model not generalising well.
Hyperparameter Tuning
 Used GridSearchCV to find optimal parameters:
Parameters Tuned:
Maximum depth
Minimum samples split
Minimum samples leaf
Criterion (gini, entropy)
The best model had a balanced depth and leaf parameters, reducing overfitting.
Evaluation
Improved Test Accuracy: The tuned model performed better on unseen data.
Visualisation: The decision tree was plotted to interpret the decision rules.
Random Forest Classifier
Initial Model
Training a random forest with default parameters resulted in:
Low Test Accuracy (23%): When training a random forest with default parameters, the model resulted in low test accuracy (23%). This poor performance was likely due to potential overfitting and insufficient trees in the ensemble.
Improved Model
To increase the performance several parameters were adjusted:
Increased Number of Trees: Set n_estimators to 100.
Random State: Ensured reproducibility.
The updated model showed:
High Test Accuracy (90%): Indicating that ensemble methods improved performance.
Support Vector Machine (SVM)
Trained an SVM model with:
Kernel: Radial Basis Function (rbf).
Max Iterations: Increased to 10,000 for convergence.
The SVM model achieved:
High Training and Testing Accuracy: Suggesting good generalisation.
XGBoost Classifier
Implemented an XGBoost classifier with adjusted parameters to prevent overfitting:
Learning Rate: Set to 0.01 for gradual learning.
Max Depth: Limited to 5.
Regularisation: Applied L1 and L2 regularisation.
The model showed:
Good Training Accuracy
Good Testing Accuracy
Results and Discussion:
Through iterative modelling and tuning, we observed the following:
Logistic Regression: Required regularisation and class balancing to perform well.
Model Scores : Training Accuracy - 0.89 or 89% and Testing accuracy - 0.89 or 89%
Decision Tree: Prone to overfitting without parameter tuning.
Model Scores : Training Accuracy - 0.9789 or 97.89% and Testing Accuracy - 0.8963 or 89.63%
Random Forest: Initial poor performance improved significantly by increasing the number of trees and adjusting other parameters.
Model Scores : Training Accuracy - 0.97 or 97% and Testing Accuracy - 0.90 or 90%
SVM and XGBoost: Both models performed well with proper parameter settings, indicating their robustness.
SVM: Training Accuracy - 0.891 or 89.1% and Testing Accuracy - 0.8924 or 89.24%
XGBoost : Training Accuracy 0. 9183 or 91.83% and Testing Accuracy - 0.8928 or 89.28%
Feature Importance
We extracted feature importances to understand which features contributed most to the predictions.
Failures and Mistakes
Logistic Regression
Issue: Low training accuracy with high test accuracy suggested overfitting or data leakage.
Resolution:
Applied feature scaling using StandardScaler.
Tuned hyperparameters with GridSearchCV.
Set class_weight to 'balanced' to handle class imbalance.
Resulted in consistent performance across training and testing sets.
Random Forest Classifier
Issue: Achieved only 23% test accuracy initially.
Cause: Default parameters were insufficient, and the model might have been underfitting.
Resolution:
Increased n_estimators to 100 to allow the model to capture more patterns.
Tuned other hyperparameters like max_depth, min_samples_split, and min_samples_leaf.
The improved model achieved 90% test accuracy.
Data Splitting Error
Issue: Initially, there was a mismatch in the shapes of x_train, y_train, x_test, and y_test.
Cause: Incorrect order of parameters in train_test_split.
Resolution:
Corrected the order to X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2).
Ensured that features and targets were properly aligned.
Outlier Handling
Issue: Outliers were affecting model performance.
Resolution:
Used the Z-score method to remove outliers beyond 3 standard deviations.
Resulted in a cleaner dataset and improved model accuracy.
Conclusion:
By systematically exploring, visualising, and preprocessing the data, and by carefully tuning the models, a good accuracy in classifying weather types was achieved. The process highlighted the importance of:
Data Cleaning: Removing duplicates and outliers.
Feature Engineering: Encoding categorical variables and selecting relevant features.
Model Selection: Trying different algorithms to find the best fit.
Hyperparameter Tuning: Using techniques like grid search to optimise model performance.
Validation: Employing cross-validation and evaluating with appropriate metrics.

