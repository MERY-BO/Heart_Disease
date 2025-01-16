# Heart Disease Prediction with Machine Learning

## Project Overview
This project aims to predict heart disease using various machine learning models. The dataset used contains clinical and demographic features related to heart health. The accuracy achieved by the best-performing model is **98%**.

## Dataset
The dataset used for this project is `heart.csv`. It contains the following features:

- **Age**: Age of the patient
- **Sex**: Gender of the patient
- **Chest Pain Type**: Type of chest pain experienced
- **Resting Blood Pressure**: Resting blood pressure (in mm Hg)
- **Cholesterol**: Serum cholesterol (in mg/dl)
- **Fasting Blood Sugar**: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
- **Resting ECG**: Resting electrocardiographic results
- **Max Heart Rate**: Maximum heart rate achieved
- **Exercise Induced Angina**: Exercise-induced angina (1 = yes; 0 = no)
- **Oldpeak**: ST depression induced by exercise relative to rest
- **Slope**: Slope of the peak exercise ST segment
- **Number of Major Vessels**: Number of major vessels (0-3) colored by fluoroscopy
- **Thalassemia**: A blood disorder type (normal, fixed defect, reversible defect)
- **Target**: The presence of heart disease (1 = disease; 0 = no disease)

## Machine Learning Models
The following models were implemented:

### Scale-Insensitive Models
1. **Random Forest Classifier**
2. **Naive Bayes Classifier (GaussianNB)**
3. **Gradient Boosting Classifier**

### Scale-Sensitive Models
1. **K-Nearest Neighbors (KNN)**
2. **Logistic Regression**
3. **Support Vector Classifier (SVC)**

### Model Evaluation Metrics
- Accuracy
- Recall Score
- Confusion Matrix
- ROC-AUC Curve

## Steps Performed
1. **Data Preparation**:
   - Split the dataset into training and testing sets using `train_test_split`.
   - Standardized the scale-sensitive models with `StandardScaler`.

2. **Model Training**:
   - Trained both scale-sensitive and scale-insensitive models on the training data.

3. **Model Evaluation**:
   - Evaluated models using accuracy, recall, and confusion matrices.
   - Plotted the ROC curve and calculated the AUC for the Random Forest model.

4. **Hyperparameter Tuning**:
   - Optimized the Random Forest model using `GridSearchCV` for better performance.

5. **Feature Importance Analysis**:
   - Visualized feature importances for the best-performing model.

6. **Correlation Analysis**:
   - Generated a heatmap to show correlations between features.

## Results
- **Best Model**: Random Forest Classifier
- **Accuracy**: 98%

## Visualizations
1. **ROC Curve**: Receiver Operating Characteristic curve for Random Forest Classifier.
2. **Feature Importance**: Bar plot of feature importances from the optimized Random Forest model.
3. **Correlation Heatmap**: Heatmap showing correlations between features.


## Libraries Used
- `pandas`
- `matplotlib`
- `seaborn`
- `numpy`
- `scikit-learn`

## Future Work
- Explore additional algorithms like XGBoost or CatBoost for improved performance.
- Integrate the model into a web application using Flask or FastAPI for real-time predictions.


## Acknowledgements
Special thanks to the creators of the dataset and the open-source community for their resources.
