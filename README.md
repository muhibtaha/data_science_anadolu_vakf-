# Kodluyoruz Data Science & Machine Learning Bootcamp - My Journey & Final Project

## About This Repository

Welcome to my repository for the Kodluyoruz Data Science & Machine Learning Bootcamp! This space documents my learning journey, key concepts covered, and the final project I developed as a culmination of the 8-week intensive program. The bootcamp provided a comprehensive introduction to the world of data science, covering everything from foundational algorithms and Python programming to advanced machine learning techniques.

## Bootcamp Overview (8 Weeks)

The bootcamp was structured to build a strong foundation and progressively introduce more complex topics. Key areas of learning included:

1.  **Week 1-2: Foundations & Python Programming**
    *   Introduction to algorithms and computational thinking.
    *   Core Python programming: data types, control flow (loops, conditionals), functions, and object-oriented programming (OOP) basics.
    *   Essential Python libraries for data science: NumPy for numerical operations and pandas for data manipulation and analysis.

2.  **Week 3-4: Data Analysis & Visualization**
    *   Data cleaning and preprocessing techniques: handling missing values, data type conversions, outlier detection.
    *   Exploratory Data Analysis (EDA): Using pandas to summarize, filter, group, and transform data.
    *   Data visualization: Creating insightful charts and graphs using Matplotlib and Seaborn to understand data distributions, relationships, and trends.

3.  **Week 5-6: Introduction to Machine Learning**
    *   Fundamental concepts of machine learning: supervised vs. unsupervised learning, regression vs. classification.
    *   Understanding the machine learning workflow: data collection, preprocessing, feature engineering, model selection, training, and evaluation.
    *   Introduction to key algorithms:
        *   **Regression**: Linear Regression, Decision Trees.
        *   **Classification**: Logistic Regression, K-Nearest Neighbors (KNN), Decision Trees.
    *   Model evaluation metrics: Accuracy, Precision, Recall, F1-Score, Mean Squared Error (MSE), R-Squared.
    *   The importance of splitting data: training, validation, and testing sets.

4.  **Week 7-8: Advanced Topics & Project Work**
    *   More advanced machine learning models: Random Forests, and an overview of other ensemble methods.
    *   Feature engineering and selection techniques.
    *   Introduction to hyperparameter tuning.
    *   Focus on developing a capstone project to apply learned skills.

## Key Skills Acquired

*   **Algorithmic Thinking**: Ability to break down problems and design efficient solutions.
*   **Python Programming**: Proficiency in Python for data analysis and machine learning tasks.
*   **Data Wrangling & Preprocessing**: Cleaning, transforming, and preparing data for analysis and modeling.
*   **Exploratory Data Analysis (EDA)**: Uncovering patterns, anomalies, and insights from data.
*   **Data Visualization**: Communicating data stories effectively through visual means.
*   **Machine Learning Modeling**: Selecting, training, and evaluating various machine learning models for regression and classification tasks.
*   **Model Evaluation & Interpretation**: Assessing model performance and understanding feature importance.
*   **Problem Solving**: Applying data science techniques to solve real-world (or representative) problems.

## Final Project: CO2 Emissions Prediction

As the final project for this bootcamp, I developed a machine learning model to predict CO2 emissions (in kilotons) for different countries.

*   **Objective**: To build a regression model that accurately estimates CO2 emissions based on various socio-economic and energy-related indicators.
*   **Dataset**: "Global Data on Sustainable Energy" from Kaggle, containing features like access to electricity, renewable energy capacity, GDP, and more.
*   **Methodology**:
    1.  Downloaded and explored the dataset using `pandas`.
    2.  Preprocessed the data by handling missing values (mean imputation for numerical features) and selecting relevant features.
    3.  Split the data into training and testing sets.
    4.  Trained a `DecisionTreeRegressor` model.
    5.  Evaluated the model using R-squared, MSE, and MAE. The model achieved an R-squared score of approximately 0.993.
    6.  Analyzed feature importances, identifying "Electricity from fossil fuels (TWh)" as a key predictor.
    7.  Performed basic hyperparameter tuning by testing different `max_depth` values for the decision tree, further improving the R2 score slightly to ~0.9932 with `max_depth=15`.
*   **Outcome**: The project successfully demonstrated an end-to-end machine learning workflow, resulting in a model with high predictive accuracy for CO2 emissions.

This bootcamp has been an incredible learning experience, equipping me with the foundational knowledge and practical skills to pursue a career in data science and machine learning. I am excited to continue learning and applying these skills to new challenges.

---

*This README was created to summarize my participation and work in the Kodluyoruz Data Science & Machine Learning Bootcamp.*
