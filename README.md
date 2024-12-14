# Income Prediction Model

This project is designed to predict income levels based on various demographic and employment features. The dataset used is the "Adult Income" dataset, which categorizes individuals based on their income level (<=50K or >50K). The task is to predict whether a person earns more than 50K based on various features.

## Features of the Project

### Data Preprocessing:
- Replaces missing values (?) with NA and drops rows containing these missing values.
- Maps categorical variables to numerical values (e.g., "Male" → 0, "Female" → 1).
- Applies one-hot encoding to categorical columns like "Work Class", "Education", "Marital Status", etc.

### Model Building:
Various machine learning models are applied, including:
- K-Nearest Neighbors (KNN)
- Decision Tree Classifiers (with different criteria: gini, entropy, log_loss)
- Logistic Regression
- Support Vector Machines (SVM) with multiple kernels (linear, poly, rbf, sigmoid)
- Naive Bayes
- Random Forests

### Model Evaluation:
- Models are evaluated using accuracy, confusion matrix, and classification reports.
- The dataset is split into training and testing sets for performance evaluation.

### Data Balancing:
- Balances the dataset using oversampling of the minority class and undersampling of the majority class.

## Running:
You can run the notebook on Google Colab for easier setup. Simply open the .ipynb file directly in Google Colab.

## Dataset

The dataset is automatically downloaded from Google Drive using the `gdown` library in the notebook.

```python
file_id = "1IzivFNReeDrOffY2QZMaQ0X9ineEqTBv"
url = f"https://drive.google.com/uc?id={file_id}"
gdown.download(url, "adult.csv", quiet=False)
```

## How to Use

1. Open the Jupyter notebook (`income_prediction.ipynb`) in Google Colab or Jupyter Notebook.
2. Run each cell sequentially to load, preprocess the data, and train the machine learning models.
3. Evaluate the models' performance with metrics such as accuracy, confusion matrix, and classification report.

## Model Training

The following models are trained and evaluated in the notebook:

- KNN (K-Nearest Neighbors)
- Decision Tree Classifiers
- Logistic Regression
- Support Vector Machine (SVM)
- Naive Bayes
- Random Forest

### Evaluation Metrics:
Each model’s performance is evaluated with:
- Training Accuracy
- Testing Accuracy
- Confusion Matrix
- Classification Report

## Results

The evaluation results (accuracy, confusion matrix, and classification reports) for each model are displayed within the notebook. You can use these results to compare the performance of different models and choose the most suitable one.
