# Lung Cancer Detection 
#  DATA_606_UMBC Data Science Capstone Project- Chaojie Wang
**Author:** Swapan Gupta Chollati\
**Semester:** Spring 2023

[Lung Cancer Detection Presentation link](https://github.com/SwapanGupt/SwapanGupta_Data606/blob/main/docs/Lung%20Cancer%20Detection.pptx)

[Youtube Link]()

# Introduction
Every year, millions of people die from lung cancer, making early detection is essential for effective treatment. The number of individuals experiencing issues with their lungs is rising today and the number of people who are susceptible to lung cancer is growing for a variety of reasons. 

However, it is costly and difficult to perform cancer tests to all these people. So a model which can detect lung cancer just by inputting few identifiable things would reduce the cost a lot by reducing the number of patients to test. By leveraging advanced techniques in deep learning, we hope to create a tool that can help doctors make more informed decisions and improve patient outcomes
# Data Collection
The data has been collected from the below link:
https://www.kaggle.com/code/sripadkarthik/lung-cancer-prediction-using-ml-and-dl/data

![Image](https://github.com/SwapanGupt/SwapanGupta_Data606/blob/main/images/Dataset.png?raw=true)

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
![Image](https://github.com/SwapanGupt/SwapanGupta_Data606/blob/main/images/Label%20Distribution.png?raw=true)

Here, the data is almost equally distributed, This is good for using classification models.
## Distribution of Patient Ages
![Image](https://github.com/SwapanGupt/SwapanGupta_Data606/blob/main/images/Distribution%20of%20Patient%20Ages.png?raw=true)

Here, we can see that age group between 33 to 36 are mostly affected by lung cancer.
## Scatter plot of the features:
![Image](https://github.com/SwapanGupt/SwapanGupta_Data606/blob/main/images/Scatter%20Plot.png?raw=true)

Here, we can see how the patients are distriuted and also notice the decision boundary. Some medium and low risk patients are very near. However this could happen after using PCA which results in loss of information and hopefully we can achieve to set a good boundary after apply ML models on more than 2D level
# Preprocessing:
For further analysis we need to preprocess the data. In our data we can observe that all the values are positives and there are no string values.
## Scaling:
MinMax scaling is a data preprocessing technique used to transform numeric data into a specific range, typically between 0 and 1. 
This ensures that all data points are scaled proportionally to each other, without distorting the relative differences between them.
![image](https://user-images.githubusercontent.com/89816922/236922099-88a0ea19-ad12-4593-9ede-66903ac2a64b.png)
After normalizing we can observe that the data is scaled in between 0 and 1.
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
- K-nearest neighbors (KNN) is a type of supervised learning algorithm.
- It makes predictions based on the similarity of input data to labeled examples in a training dataset.
- The algorithm records all of the labels and training examples during training.
- KNN locates the k nearest neighbors in the training data and gives the majority label among them to the input to create a prediction for a new input.
KNN has been performed with different values for k and the respective accuracy values are plotted as below:
![image](https://github.com/SwapanGupt/SwapanGupta_Data606/blob/main/images/KNN.png?raw=true)
- We can see how accuracy increased until k is 3 and decreased after k is 7. Similar case was oserved with different proportions of testsets with different random states. So taking k as 5 which was the mean in many cases.
- By taking k as 5 validations are performed and the results are given in results section.
## Confusion Matrix for KNN
![image](https://github.com/SwapanGupt/SwapanGupta_Data606/blob/main/images/KNN%20Low.png?raw=true)
![image](https://user-images.githubusercontent.com/89816922/236923025-7689e029-062d-483a-9218-cecaafd450a4.png)
![image](https://github.com/SwapanGupt/SwapanGupta_Data606/blob/main/images/KNN%20High.png?raw=true)

## Decision tree
- Decision tree is a supervised learning algorithm used for multi-label classification tasks.
- It builds a tree-like model to classify the input data based on the features.
- The tree is constructed recursively by splitting the input space based on the feature values that maximize the information gain or minimize the impurity.
Each leaf node of the tree represents a label combination, and the path from the root to a leaf node determines the label assignment
The obtained decision tree was
![image](https://user-images.githubusercontent.com/89816922/236923273-0b5482f0-1ef1-46b0-ab16-a25f64f2a650.png)
### Conditions of Decision Tree
![image](https://github.com/SwapanGupt/SwapanGupta_Data606/blob/main/images/DT%20Conditions.png?raw=true)

## Confusion Matrix of Decision Tree
![image](https://github.com/SwapanGupt/SwapanGupta_Data606/blob/main/images/DT%20Low.png?raw=true)
![image](https://user-images.githubusercontent.com/89816922/236923782-cef4b76a-dfa6-4a17-956c-769e7c3c6949.png)
![image](https://github.com/SwapanGupt/SwapanGupta_Data606/blob/main/images/DT%20High.png?raw=true)

## Logistic regression
- Logistic regression is a supervised learning algorithm used for binary or multi-class classification tasks.
- It models the relationship between the input features and the probability of the output label.
- The approach calculates the input feature coefficients that maximize the likelihood of the labels that were observed in the training set.
The likelihood of each class is calculated during prediction, and the output label is given to the class with the highest probability.
We can observe the summary of the model as below:
![image](https://user-images.githubusercontent.com/89816922/236924073-5d585645-1125-451f-be94-559684b28d9d.png)
## Loss and Accuracy curves
![image](https://user-images.githubusercontent.com/89816922/236924137-0003857a-032e-4d5d-8046-ca1deeb530ff.png)
## Confusion Matrix of Logistic Regression
![image](https://user-images.githubusercontent.com/89816922/236924691-ae72182c-cede-43bc-93ad-5d7faa98ba3f.png)
![image](https://user-images.githubusercontent.com/89816922/236924743-5dd37221-66d9-44cb-94eb-023f12aa4969.png)
![image](https://user-images.githubusercontent.com/89816922/236924779-8a6853f9-0ba6-48bd-8dac-5bd75f8d4b60.png)
This is a case where we should be alarming as many patients as possible. Here, the deciding factor is False negative as it could put the patient at risk. So, we will be considering recall of high-risk patients as the main factor
As KNN and Logistic regression performed similar in this case, considering other factors like f1 score, I decided to use KNN algorithm for predictions.
# Conclusion
- In conclusion, applying automated models to detect lung cancer will reduce the efforts of doctors very much and thus the doctors will be able to concentrate on treating the people who require it.
- I have trained 3 Models KNN, Logistic Regression, Decision Tree and ou of which KNN is the best performed model.
- f1 score of KNN was about 0.99,0.90.0.92 and precision was 0.98,0.99,0.90 for High, Low and Medium respectively
- The availability and caliber of the data, as well as the thorough selection and fine-tuning of machine learning algorithms, are what determine whether such a project is successful. A lung cancer detection model could potentially save lives by detecting cancer sooner and enhancing treatment outcomes with the correct tools and knowledge.





