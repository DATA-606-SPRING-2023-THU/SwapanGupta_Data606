# UMBC Data Science Capstone Project
**Author:** Swapan Gupta Chollati\
**Semester:** Spring 2023
# Introduction
Every year, millions of people die from lung cancer, making early detection is essential for effective treatment. The number of individuals experiencing issues with their lungs is rising today and the number of people who are susceptible to lung cancer is growing for a variety of reasons. 

However, it is costly and difficult to perform cancer tests to all these people. So a model which can detect lung cancer just by inputting few identifiable things would reduce the cost a lot by reducing the number of patients to test. By leveraging advanced techniques in deep learning, we hope to create a tool that can help doctors make more informed decisions and improve patient outcomes
# Data Collection
The data has been collected from the below link:
https://www.kaggle.com/code/sripadkarthik/lung-cancer-prediction-using-ml-and-dl/data

- The unit of analysis is a patient
- The dataset has 1000 units of analysis
# Features:
Below are the variables I am considering for analysis
- Age
- Gender
- Air Pollution
- Alcohol use
- Dust Allergy
- OccuPational Hazards
- Genetic Risk
- chronic Lung Disease
- Balanced Diet
- Obesity
- Smoking
- Passive Smoker
- Chest Pain
- Coughing of Blood
- Fatigue
- Weight Loss
- Shortness of Breath
- Wheezing
- Swallowing Difficulty
- Clubbing of Finger Nails
- Frequent Cold
- Dry Cough
- Snoring

# Exploratory Data analysis:
## Label distribution:
Here, the data is almost equally distributed, This is good for using classification models.

## Scatter plot of the features:

# Preprocessing:
## Scaling:
MinMax scaling is a data preprocessing technique used to transform numeric data into a specific range, typically between 0 and 1. 
This ensures that all data points are scaled proportionally to each other, without distorting the relative differences between them.
## One Hot Encoding:
One hot encoding is a process of converting categorical information into a format that can be used by Machine learning algorithms.
The target variable contains 3 still values low, medium and high.
The 3 values are vectorized as Low-100, Medium-010, High-001 and the returned sparse matrix is converted to dense matrix by using todense function.
# Models:
Models are trained and analyzed
- K- nearest neighbors
- Decision tree
- Logistic regression
The confusion matrix is drawn for analysis

## K - nearest neighbors:




