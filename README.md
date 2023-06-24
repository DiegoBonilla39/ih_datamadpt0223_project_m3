
# 💎💰 Supervised Learning Project - Predict Diamond Prices

## 🖹 Description

- The objective of the project is to **predict prices for a specific set of diamonds** by training supervised learning regression models with a training dataset. 
- The setting of the project was through a Kaggle competition. More information about the competition can be found here: https://www.kaggle.com/competitions/ihdatamadpt0223projectm3
- The evaluation metric chosen for this competition is the **RMSE** (Root Mean Squared Error)

### ⭕ Diamonds Features

- **price**: price in USD
- **carat**: weight of the diamond
- **cut**: quality of the cut (Fair, Good, Very Good, Premium, Ideal)
- **color**: diamond colour, from J (worst) to D (best)
- **clarity**: a measurement of how clear the diamond is (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best))
- **x**: length in mm
- **y**: width in mm
- **z**: depth in mm
- **depth**: total depth percentage = z / mean(x, y) = 2 * z / (x + y) (43--79)
- **table**: width of top of diamond relative to widest point (43--95)
- **city**: city where the diamonds is reported to be sold.

### 🔢 Data Sources

- diamonds_train.csv
- diamonds_train.db
- diamonds_predict.csv
- sample_submission.csv

## 📁 Folder Structure

```bash
└── project   
    └── data
        ├── diamonds_train.csv
        ├── diamonds_train.db
        ├── diamonds_predict.csv
        └── sample_submission.csv
    ├── final notebooks
        ├── model_1.ipynb
        └── model_2.ipynb
    ├── final submissions
        ├── model_1.csv
        └── model_2.csv
    ├── models
        ├── model_1.sav
        └── model_2.sav
    ├── .gitignore
    └── README.md
```
    
## 💻 Technology Stack

- pandas
- numpy
- matplotplib
- seaborn
- sklearn
- pickle

## ✔️ Step-By-Step

- Exploratory Data Analysis of the dataframes
- Data preparation for training
    - Feature engineering
    - Data filtering
    - Encoding of categorical features
    - Scaling of numerical features
    - Feature selection
- Model definition
    - Model selection
    - Hyperparameters Tuning
    - Continuous trial and error process
- Model application
    - Training with the whole training dataset
    - Obtain predictions of diamond prices

##  🤖 Models 

###  🧠 Used Models for Training

- sklearn.neighbors.KNeighborsRegressor
- sklearn.linear_model.Lasso
- sklearn.tree.DecisionTreeRegressor
- sklearn.ensemble.ExtraTreesRegressor
- sklearn.ensemble.RandomForestRegressor
- sklearn.ensemble.HistGradientBoostingRegressor
- sklearn.tree.GradientBoostingRegressor

### ⭐ Final Models (Best Scores)

| Model                           | model 1                   | model 2                   |
|---------------------------------|---------------------------|---------------------------|
| Training Data                   | diamonds = diamonds_train.csv + diamonds_test.csv | diamonds = diamonds_train.csv + diamonds_test.csv |
| Feature Engineering             | - Creation of 'x/y ratio'     | - Creation of 'x/y ratio'     |
|                                 | - Rescaling of ‘table’ and ‘depth’ | - Rescaling of ‘table’ and ‘depth’ |
| Data filtering                  | - Drop zero values for x, y and z | - Drop zero values for x, y and z |
|                                 | - Drop y outliers           | - Drop y outliers           |
|                                 | - Drop z outliers           | - Drop z outliers           |
| Encoding                        | Label encoding of categorical features | Label encoding of categorical features |
| Scaling                         | Robust scaling of numerical features | Robust scaling of numerical features |
| Feature Selection               | - carat                     | - carat                     |
|                                 | - cut                       | - cut                       |
|                                 | - color                     | - color                     |
|                                 | - clarity                   | - clarity                   |
|                                 | - depth (%)                | - depth (%)                |
|                                 | - table (%)                | - table (%)                |
|                                 | - x/y ratio                 | - x/y ratio                 |
| Selected Model and Hyperparameters | GradientBoostingRegressor | GradientBoostingRegressor |
|                                 | - random_state = 42         | - random_state = 42         |
|                                 | - learning_rate = 0.01      | - learning_rate = 0.01      |
|                                 | - max_depth = 7             | - max_depth = 7             |
|                                 | - n_estimators = 1000       | - n_estimators = 1000       |
|                                 | - subsample = 0.8           | - subsample = 0.78           |
| Grid Search CV                  | 508.71211       | 508.36514        |
| RMSE - train_test_split         | 518.64008        | 519.77150        |
| RMSE - test in Kaggle (50%)               | 505.04728                | 506.05712                |

## ➡️ Output

- **Predictions**: Two .csv files (model_1.csv and model_2.csv) with the submissions to upload in the Kaggle competition.
- **Models**: Two .sav files (model_1.sav and model_2.sav) with the models to import in case of needed usage.