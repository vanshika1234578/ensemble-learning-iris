# Ensemble Learning on the Iris Dataset

A mini project that implements the three main ensemble learning techniques with scikit-learn and XGBoost: **stacking**, **bagging** and **boosting**. All models are trained and tested on the Iris flower dataset to classify species (setosa, versicolor, virginica).

## What is covered
- **Stacking:** Decision Tree, SVM and Logistic Regression as base learners, with a Logistic Regression meta-learner (`StackingClassifier`, `cv=5`)
- **Bagging:** Random Forest
- **Boosting:** AdaBoost, Gradient Boosting and XGBoost

The train/test split is done first, and the cross-validation inside stacking runs only on the training data. The final model is scored on the unseen test set.

## Dataset
Iris, loaded with `sns.load_dataset("iris")` (150 rows, 4 numeric features). The first run needs internet access to download it. Labels are converted with `LabelEncoder` and the split is 80/20 (`random_state=42`).

## Results (test set, 30 samples)
| Technique | Model | Accuracy |
|-----------|-------|----------|
| Stacking | DT + SVM + LR, LR meta-learner | 1.000 |
| Bagging | Random Forest | 1.000 |
| Boosting | AdaBoost | 0.933 |
| Boosting | Gradient Boosting | 1.000 |
| Boosting | XGBoost | 1.000 |

Iris is small and easy to separate, and the test set has only 30 flowers, so near-perfect scores are expected. Cross-validation gives a more reliable estimate.

## Tech stack
Python, pandas, seaborn, scikit-learn, XGBoost, Jupyter

## How to run
    git clone https://github.com/vanshika1234578/ensemble-learning-iris.git
    cd ensemble-learning-iris
    pip install -r requirements.txt
    jupyter notebook ensemble_learning_iris.ipynb
