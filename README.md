# task2:Machine-Learning-Model

# TASK2 : Machine-Learning-Model

## Table of Contents⚙️
- Project Description
- Technologies Used
- Code Explanation
- File Structure

## Project Description📝

This project use machine learning to create a model that predicts which passengers survived the Titanic or not Using Logistic Regression to train the model, and accuracy score to evaluate the model.

## Technologies Used 🔧

- Google colab
- Kaggle
- Excel
- Python

## 💻 Code Explanation

- Go to Google Colab:
    https://colab.research.google.com/

- Uploading files:
    - From Open notebook choose Upload to upload Titanic_ML(task)S(1).ipynb
    - From Files choose Upload to upload train.cvs

- We only wrote the modified codes.

1. importing the Dependencies
2. Reading the data

3. Data Preprocessing
- Dealing with Missing Data:

You have three options to fix this:

Delete rows that contains missing valuse
Delete the whole column that contains missing values
Replace missing values with some value (Mean, Median, Mode, constant)

There are three columns contains Missing values: Age, Cabin, Embarked.
In the Age column, we will fill the missing values with the mean since it is a simple and quick method to handle missing data and helps maintain the overall distribution of the dataset.
#fill the missing values in Age with the mean of Age column
#you can simply use 'filllna' function, or any other way such as SimpleImputer
import pandas as pd
mean_age = data['Age'].mean()
data['Age'].fillna(mean_age, inplace=True)
#fill the missing values in Embarked with the mode of Embarked column:
mode_embarked = data['Embarked'].mode()[0]
data['Embarked'].fillna(mode_embarked, inplace=True)
- Drop useless columns:
# Drop the PassengerId and Name Columns from the dataset:
data = data.drop(['PassengerId', 'Name'], axis=1)
print(data)
- Encode Categorical Columns
Sex and Embarked columns values are text, we can't give this text directly to the machine learning model, so we need to replace this text values to meaningful numerical values.
- Dealing with Duplicates
#check if there are duplicates in the dataset:
duplicates = data.duplicated()

# Print the rows that are duplicates(there are no duplicates coulmn)
print("Duplicate Rows:")
print(data[duplicates])
if duplicates.any():
    print("There are duplicates in the dataset")
else:
    print("There are no duplicates in the dataset")
#drop the duplicates:
data = data.drop_duplicates()
- Data Analysis: In this section, we will explore the data and the relationships between features using statistical analysis and visualization techniques. This will help us understand the underlying patterns and correlations in the dataset, providing valuable insights for model building.
4. Model Building
from sklearn.model_selection import train_test_split

# Split the data into training data & Testing data using train_test_split function :
# Assuming 'Age' is the feature and 'Embarked' is the target
X = data[['Age']]  # Features (input variables)
y = data['Embarked']  # Target (output variable)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Training Features:")
print(X_train)
print("\nTesting Features:")
print(X_test)
print("\nTraining Target:")
print(y_train)
print("\nTesting Target:")
print(y_test)
-Model Training is a crucial step in the machine learning where the algorithm learns from the training data to make predictions. Logistic Regression is a commonly used algorithm for binary classification tasks, such as predicting whether a passenger survived in the Titanic dataset. By training the model on our training data, we aim to find the best-fit parameters that minimize prediction errors. Once trained, this model can be used to predict outcomes on new, unseen data.
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

