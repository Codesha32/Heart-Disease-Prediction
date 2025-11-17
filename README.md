# Heart-Disease-Prediction using Machine Learning

## Overview
This project implements a machine learning pipeline for predicting heart disease based on the UCI Heart Disease dataset. The dataset contains patient medical attributes, and the goal is to classify whether a patient has heart disease (`target = 1`) or not (`target = 0`).

We explore multiple classifiers, evaluate their performance using accuracy and recall, perform hyperparameter tuning on the Random Forest model, and visualize key insights like ROC curves, feature importances, and correlation heatmaps.

This code is designed to run in a Jupyter Notebook environment.

## Dataset
- **Source**: UCI Machine Learning Repository (loaded from `heart.csv`).
- **Shape**: 1025 rows × 14 columns.
- **Features** (13 input features):
  - `age`: Age of the patient.
  - `sex`: Sex (1 = male, 0 = female).
  - `cp`: Chest pain type (1-4).
  - `trestbps`: Resting blood pressure.
  - `chol`: Serum cholesterol (mg/dl).
  - `fbs`: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false).
  - `restecg`: Resting electrocardiographic results (0-2).
  - `thalach`: Maximum heart rate achieved.
  - `exang`: Exercise-induced angina (1 = yes, 0 = no).
  - `oldpeak`: ST depression induced by exercise.
  - `slope`: Slope of the peak exercise ST segment (1-3).
  - `ca`: Number of major vessels colored by fluoroscopy (0-3).
  - `thal`: Thalassemia (3 = normal, 6 = fixed defect, 7 = reversible defect).
- **Target**: Presence of heart disease (1 = disease, 0 = no disease).
- **Sample Data Preview**:
  ```
  age  sex  cp  trestbps  chol  fbs  restecg  thalach  exang  oldpeak  slope  ca  thal  target
  0    52   1   0     125     212    0      1       168     0        1.0      2   2     3       2
  1    53   1   0     140     203    1      0       155     1        3.1      0   3     0       1
  ...
  ```

## Requirements
- Python 3.8+
- Libraries: `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `numpy`

Install via pip:
```
pip install pandas scikit-learn matplotlib seaborn numpy
```

## Usage
1. **Load and Prepare Data**:
   - Run the initial cells to load `heart.csv` and split into train/test sets (60/40 split, `random_state=9`).

2. **Train Models**:
   - **Scale-Insensitive Models**: Random Forest, Naive Bayes, Gradient Boosting.
   - **Scale-Sensitive Models**: K-Nearest Neighbors (KNN), Logistic Regression, Support Vector Classifier (SVC) – applied after StandardScaler.

3. **Evaluate Models**:
   - Accuracy and recall scores are computed on the test set.
   - ROC curve and AUC for Random Forest.

4. **Hyperparameter Tuning**:
   - Grid search on Random Forest for `n_estimators` (values: 100, 200, 500, 600, 700) using 3-fold CV.

5. **Visualizations**:
   - ROC Curve (Random Forest).
   - Feature Importances (horizontal bar plot for tuned Random Forest).
   - Correlation Heatmap (full dataset).

To run the full notebook:
- Place `heart.csv` in the same directory.
- Execute cells sequentially in Jupyter.

## Models and Performance
### Trained Classifiers
| Model                  | Scaling Required? | Key Parameters                  | Accuracy | Recall  |
|------------------------|-------------------|---------------------------------|----------|---------|
| Random Forest         | No                | Default (`n_estimators=100`)   | 0.985   | 0.986  |
| Naive Bayes           | No                | Default (GaussianNB)            | 0.846   | 0.901  |
| Gradient Boosting     | No                | Default                         | 0.971   | 0.986  |
| K-Nearest Neighbors   | Yes               | Default (`n_neighbors=5`)       | 0.854   | 0.873  |
| Logistic Regression   | Yes               | Default                         | 0.873   | 0.920  |
| Support Vector Machine| Yes               | Default (SVC kernel='rbf')      | 0.934   | 0.953  |

- **Best Model (Untuned)**: Random Forest (highest accuracy and recall).
- **ROC AUC (Random Forest)**: 0.999 (excellent discrimination).

### Tuned Random Forest
- Best parameters: `n_estimators=500`.
- Improved model stored as `best_forest`.

## Visualizations
- **ROC Curve**: Plots True Positive Rate vs. False Positive Rate for Random Forest predictions.
- **Feature Importances**: Horizontal bar chart showing relative importance of each feature (sorted ascending). Uses a yellow-green colormap.
- **Correlation Heatmap**: Seaborn heatmap of feature correlations (annotated, `cmap='YlGn'`).

**Note**: The feature importance plot code in the notebook returns the `plt.show` function reference; ensure `plt.show()` is called explicitly to display.

## Key Insights
- Random Forest and Gradient Boosting perform best, likely due to handling non-linear relationships and interactions well.
- Scaling improves distance-based models (KNN, Logistic Regression, SVM).
- Top features (from tuned Random Forest): Typically `ca`, `thal`, `oldpeak`, `cp` (exact ordering varies; visualize for details).
- Dataset shows moderate correlations (e.g., `thalach` negatively correlates with `age`).

## Limitations and Next Steps
- **Imbalanced Data**: Check class distribution; consider SMOTE if needed.
- **Cross-Validation**: Expand GridSearchCV to more parameters (e.g., `max_depth`, `min_samples_split`).
- **Deployment**: Wrap best model in a Streamlit/Flask app for predictions.
- **Extensions**: Add confusion matrices, precision-recall curves, or SHAP explanations.
