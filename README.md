# West Nile Virus Prediction using Mosquito Trap Data

This project aims to predict the presence of West Nile Virus (WNV) in mosquitoes based on various environmental factors such as weather conditions, mosquito trap locations, and spraying schedules. The goal is to develop a predictive model that can be used to inform public health efforts in combating the spread of WNV.

## Project Overview

The dataset used for this project contains mosquito trap data, weather information, and spray data collected over several years. The main task is to predict whether the West Nile Virus is present in a given mosquito sample based on these environmental variables.

### Files

- `train.csv`: Training data containing mosquito trap information and WNV presence labels.
- `test.csv`: Test data to be used for making predictions.
- `weather.csv`: Weather data corresponding to the dates of mosquito trap collections.
- `spray.csv`: Data on mosquito spraying efforts on specific dates.
- `sampleSubmission.csv`: A sample submission file for the Kaggle competition format.

### Required Libraries

This project uses the following Python libraries:

- `pandas`
- `numpy`
- `seaborn`
- `matplotlib`
- `xgboost`
- `sklearn`

You can install the required libraries by running:

```bash
pip install pandas numpy seaborn matplotlib xgboost scikit-learn
```
## Workflow

### Data Preprocessing

1. **Data Loading**: The project starts by loading the training and testing data, as well as the weather and spray data.
2. **Missing Value Handling**: Various methods are used to deal with missing values in the datasets, including replacing them with monthly medians or removing columns with too many missing values.
3. **Feature Engineering**:
   - **Date Parsing**: The 'Date' feature is parsed into separate columns for `Year`, `Month`, `Day`, and `DayOfWeek`.
   - **Weather Data Aggregation**: Weather data is aggregated using a rolling window to smooth the data and create new features.
   - **Trap Location Encoding**: Categorical variables like `Species` and `Trap` are encoded using ordinal encoding to convert them into numerical formats suitable for machine learning models.
4. **Exploratory Data Analysis (EDA)**: Various visualizations are created to understand the distribution of features and their relationship with WNV presence.

### Model Development

1. **Data Scaling**: The features are scaled using `StandardScaler` to ensure all features are on the same scale.
2. **Dimensionality Reduction**: t-SNE is used for visualizing the data in 2D space to explore potential patterns.
3. **Model Selection**: The `XGBClassifier` is chosen for classification, and a grid search is performed to tune the hyperparameters. The model is evaluated using cross-validation with `roc_auc` as the scoring metric.
4. **Feature Importance**: After training the model, the most important features are extracted and visualized.
5. **Prediction**: The trained model is used to predict the presence of WNV on the test data, and the predictions are saved in the submission format.

### Output

- **Model Performance**: The best hyperparameters and model performance score are displayed.
- **Feature Importance**: A bar chart of the most important features based on the trained model.
- **Predictions**: The model's predictions are saved to `xgb_predictions2.csv` for submission.

### Results

- The project achieves a good predictive performance on the WNV dataset, with the best model configuration identified through hyperparameter tuning.
- Key features influencing the presence of WNV include environmental factors, mosquito trap locations, and weather patterns.

### Conclusion

This model can be used as a tool for public health officials to predict the presence of West Nile Virus in mosquito populations, allowing for timely interventions such as spraying or other control measures.
