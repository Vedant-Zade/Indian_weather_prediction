# Indian Weather & Air Quality Prediction

## 📌 Project Overview

This project develops a machine learning pipeline for predicting environmental conditions in India by integrating multiple real-world datasets related to **air quality, weather, traffic, and public holidays**.

The central idea of the project is that environmental conditions such as air pollution are not determined by a single factor. They are influenced by meteorological conditions, human activity, traffic intensity, and temporal patterns such as weekends and public holidays. Therefore, instead of training a model using only historical air-quality observations, this project combines multiple complementary data sources into a single machine-learning dataset.

The project follows an end-to-end data science workflow:

> **Data Collection → Data Cleaning → Data Transformation → Data Integration → Feature Engineering → Exploratory Data Analysis → Model Training → Model Evaluation → Prediction**

The final dataset contains historical observations enriched with weather, traffic, calendar, and air-quality information. These features are then used to train machine-learning models to learn the relationship between environmental conditions and the selected prediction target.

---

# 🎯 Problem Statement

Air pollution and environmental conditions vary significantly over time and are affected by several interacting factors.

Traditional prediction approaches that rely only on historical pollution measurements may not capture the influence of external variables such as:

- Temperature
- Humidity
- Atmospheric conditions
- Wind
- Rainfall
- Traffic activity
- Weekdays and weekends
- Public holidays
- Previous air-quality conditions

The objective of this project is therefore to construct a **multi-source machine-learning dataset** that combines these factors and use it to develop a predictive model.

The project addresses the following question:

> **Can historical air-quality observations, meteorological conditions, traffic information, and temporal/calendar patterns be combined to improve the prediction of environmental conditions?**

---

# 💡 Proposed Solution

The proposed solution integrates four major sources of information:

| Data Source | Information Provided | Purpose |
|---|---|---|
| **CPCB** | Air-quality observations | Historical pollution conditions |
| **Copernicus** | Weather observations | Meteorological conditions |
| **TomTom** | Traffic information | Human mobility/traffic conditions |
| **Public Holidays Dataset** | Indian public holidays | Calendar-based activity patterns |

These datasets have different structures, time formats, variables, and frequencies. The first major part of the project is therefore to clean and standardize them before integration.

After preprocessing, the datasets are joined using their common temporal and geographical information. Additional features are then engineered from the combined dataset.

The resulting dataset is used for exploratory analysis and machine-learning model development.

---

# 🏗️ Project Architecture

The overall workflow can be represented as:

```text
                    ┌─────────────────────┐
                    │       CPCB Data     │
                    │   Air Quality Data  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Data Cleaning &    │
                    │   Standardization    │
                    └──────────┬──────────┘
                               │
                               │
┌─────────────────────┐        │        ┌─────────────────────┐
│  Copernicus Weather │────────┤────────│   TomTom Traffic    │
│        Data         │        │        │        Data         │
└─────────────────────┘        │        └─────────────────────┘
                               │
                               │
                    ┌──────────▼──────────┐
                    │ Public Holiday Data │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Dataset Integration │
                    │  Date + Location    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Feature Engineering │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Exploratory Data    │
                    │      Analysis       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Train / Test Split  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Machine Learning    │
                    │       Model         │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Model Evaluation &  │
                    │     Prediction      │
                    └─────────────────────┘
```

---

# 📊 Data Sources

## 1. Central Pollution Control Board (CPCB)

The **Central Pollution Control Board (CPCB)** provides the air-quality observations used as the primary environmental dataset.

The dataset contains historical measurements collected from air-quality monitoring stations. These observations provide the pollution information required for the prediction problem.

The CPCB dataset is particularly important because it provides the historical values from which the model learns environmental patterns.

Typical air-quality variables include pollutant measurements such as:

- PM2.5
- PM10
- NO₂
- SO₂
- CO
- O₃
- Other available pollutant measurements depending on the monitoring station

The exact variables used in the final model depend on the availability and quality of observations after preprocessing.

### Role in the project

CPCB data provides:

> **Historical air-quality information → target and environmental predictors**

The data is cleaned to handle missing observations, inconsistent timestamps, duplicate records, and unsuitable values before being integrated with the other datasets.

---

## 2. Copernicus Weather Data

Meteorological information is obtained from the **Copernicus** data ecosystem.

Weather is an important component of the project because atmospheric conditions directly influence how pollutants disperse, accumulate, or remain concentrated.

Weather variables incorporated into the dataset can include variables such as:

- Temperature
- Relative humidity
- Wind-related variables
- Precipitation
- Atmospheric pressure
- Other available meteorological measurements

### Why weather data is important

For example, strong winds can disperse pollutants, while stagnant atmospheric conditions can allow pollutants to accumulate.

Similarly, rainfall can affect particulate matter concentrations by removing particles from the atmosphere.

Therefore, weather variables provide the model with information that cannot be obtained from air-quality observations alone.

### Role in the project

```text
Copernicus
     │
     ▼
Weather Conditions
     │
     ├── Temperature
     ├── Humidity
     ├── Wind
     ├── Rainfall
     └── Other atmospheric variables
```

These variables are aligned with the corresponding air-quality observations using the relevant date/location information.

---

# 3. TomTom Traffic Data

Traffic information is obtained from **TomTom**.

Traffic is incorporated because road traffic represents an important component of urban human activity and can influence air pollution, particularly through vehicle emissions.

Traffic conditions can also indirectly represent the level of daily human activity in an area.

The TomTom dataset is processed so that traffic-related information can be incorporated into the same temporal framework as the air-quality and weather datasets.

### Why traffic data is included

Traffic can influence pollution through:

- Vehicle emissions
- Traffic congestion
- Increased vehicle idling
- Increased road activity
- Differences between working days and non-working days

Therefore, traffic information provides an additional human-activity dimension to the prediction problem.

### Role in the project

```text
TomTom Traffic
       │
       ▼
Traffic Conditions
       │
       ├── Traffic-related measurements
       ├── Congestion/activity information
       └── Temporal traffic patterns
```

The processed traffic information is matched with the corresponding date/location in the integrated dataset.

---

# 4. Public Holiday Data

A separate public-holiday dataset is used to identify Indian public holidays.

This dataset is different from the other sources because it is **calendar information rather than environmental or traffic information**.

The public-holiday dataset contains public holidays but does not explicitly contain every Saturday and Sunday.

This distinction is important.

A day can therefore be classified using two separate concepts:

```text
Day of Week
     │
     ├── Monday
     ├── Tuesday
     ├── ...
     ├── Saturday
     └── Sunday

Public Holiday
     │
     ├── Yes
     └── No
```

This allows the final dataset to distinguish between:

- Normal working days
- Saturdays
- Sundays
- Public holidays
- Weekdays that happen to be public holidays

### Why public holidays matter

Human activity and traffic patterns can differ significantly on public holidays compared with normal working days.

Including a holiday indicator allows the model to capture these calendar-related changes.

---

# 🧹 Data Preparation

The four datasets originate from different sources and therefore cannot be directly combined.

Each dataset must first undergo preprocessing.

The general preprocessing workflow is:

```text
Raw Dataset
     │
     ▼
Remove/Handle Invalid Records
     │
     ▼
Handle Missing Values
     │
     ▼
Remove Duplicates
     │
     ▼
Standardize Column Names
     │
     ▼
Standardize Date/Time
     │
     ▼
Standardize Location Information
     │
     ▼
Convert Data Types
     │
     ▼
Prepared Dataset
```

---

# 🔄 Data Standardization

One of the most important parts of the project is making datasets from different sources compatible.

Different sources may represent the same concept differently.

For example:

```text
Source A:
2024-01-15

Source B:
15/01/2024

Source C:
2024-01-15 12:00:00
```

Before joining the datasets, the temporal information needs to be converted into a consistent representation.

The same principle is applied to geographical information and column names.

---

# 📅 Temporal Alignment

The datasets do not necessarily contain observations at exactly the same frequency.

For example:

- Air-quality data may contain frequent observations.
- Weather data may contain measurements at different temporal intervals.
- Traffic data may contain traffic-specific observations.
- Holiday data contains individual calendar dates.

To combine them into one machine-learning dataset, the information needs to be brought into a common temporal representation.

The project therefore creates a standardized **date-based representation** that can be used to associate observations from the different sources.

The final integrated structure can conceptually be represented as:

```text
Date
  │
  ├── Air Quality
  ├── Weather
  ├── Traffic
  ├── Holiday Status
  └── Day of Week
```

---

# 📍 Geographic Alignment

The data sources can contain different geographic identifiers.

For example, one source may identify an observation using a monitoring station while another may represent a geographic area using coordinates or a city/location name.

The project therefore standardizes the relevant location information before integration.

The goal is to ensure that an environmental observation is associated with the appropriate weather and traffic information rather than simply matching records based on date alone.

---

# 🔗 Dataset Integration

After preprocessing the individual datasets, they are combined into a single dataset.

The integration process can be represented as:

```text
CPCB
  │
  ├──────────────┐
  │              │
  ▼              ▼
Weather        Traffic
  │              │
  └──────┬───────┘
         │
         ▼
   Date / Location
         │
         ▼
Public Holidays
         │
         ▼
Final Integrated Dataset
```

The purpose of this step is to create one row containing as much relevant information as possible for a particular observation.

A conceptual row may therefore contain:

| Date | Location | Air Quality | Weather | Traffic | Holiday | Day |
|---|---|---|---|---|---|---|
| Date X | Location X | AQ values | Weather values | Traffic values | Yes/No | Monday |

The actual columns depend on the variables retained during preprocessing.

---

# 🧪 Feature Engineering

Raw variables are not always sufficient for machine learning.

Additional variables are therefore created from the existing information.

Feature engineering allows the model to represent patterns that are implicit in the original data.

---

## Day-of-Week Feature

A day-of-week feature is derived from the date.

For example:

```text
Monday    → 0
Tuesday   → 1
Wednesday → 2
Thursday  → 3
Friday    → 4
Saturday  → 5
Sunday    → 6
```

The exact numerical encoding depends on the implementation.

This feature helps the model identify systematic weekly patterns.

---

## Weekend Feature

A weekend indicator can be derived from the day of week.

```text
Monday–Friday → Weekday
Saturday/Sunday → Weekend
```

This is important because traffic and human activity can differ substantially between weekdays and weekends.

---

## Public Holiday Feature

The public-holiday dataset is used to create a holiday indicator.

Conceptually:

```text
Public Holiday = 1
Normal Day     = 0
```

This allows the model to distinguish public holidays from normal calendar days.

---

## Calendar Features

The date can also be used to derive additional temporal characteristics where applicable, such as:

- Day
- Month
- Year
- Day of week
- Weekend status
- Holiday status

These features allow the model to identify seasonal and temporal patterns.

---

# 🧠 Why Feature Engineering Matters

Consider two observations with identical weather conditions.

One occurs on a normal Tuesday and the other occurs on a public holiday.

The traffic and human activity may be very different.

Therefore:

```text
Weather
   +
Traffic
   +
Day of Week
   +
Holiday
   +
Historical Air Quality
   ↓
More Complete Environmental Representation
```

The purpose of feature engineering is to provide the model with this additional context.

---

# 🔎 Exploratory Data Analysis

Before model training, exploratory data analysis is performed to understand the integrated dataset.

EDA is used to answer questions such as:

- How are pollution values distributed?
- Which variables are strongly correlated?
- Are there missing values?
- Are there outliers?
- How does pollution vary over time?
- How does pollution differ between weekdays and weekends?
- Does traffic appear to be associated with pollution?
- How do weather conditions relate to pollution?
- Are there seasonal patterns?

---

# 📈 Important Graphs and Tables

The analysis can include the following visualizations.

### 1. Time-Series Plot

Shows how the target environmental variable changes over time.

```text
Date ───────────────────────────────►
        Pollution / Target
```

This helps identify:

- Long-term trends
- Sudden changes
- Seasonal behavior
- Anomalies

---

### 2. Correlation Heatmap

A correlation matrix can be used to understand relationships between numerical variables.

Example structure:

```text
             AQ   Temp  Humidity  Wind  Traffic
AQ           1
Temp              1
Humidity                1
Wind                         1
Traffic                             1
```

This is particularly useful for understanding which features may have predictive relationships with the target.

---

### 3. Distribution Plots

Distribution plots help identify:

- Skewed variables
- Outliers
- Concentration of observations
- Unusual values

---

### 4. Weekday vs Weekend Analysis

Pollution and traffic can be compared between:

```text
Weekday
   vs
Weekend
```

This provides evidence for including calendar-based features.

---

### 5. Holiday vs Non-Holiday Analysis

Similarly, environmental conditions can be compared between:

```text
Public Holiday
       vs
Normal Day
```

This helps determine whether holiday information contributes useful predictive information.

---

### 6. Weather vs Pollution Analysis

Scatter plots or similar visualizations can be used to examine relationships between weather variables and pollution.

For example:

```text
Temperature → Pollution
Humidity    → Pollution
Wind        → Pollution
Rainfall    → Pollution
```

---

### 7. Traffic vs Pollution Analysis

Traffic-related variables can be compared against pollution levels to investigate whether higher traffic activity corresponds with changes in air quality.

---

# 🧹 Missing Values

Real-world datasets commonly contain missing observations.

Missing values can occur because of:

- Sensor failure
- Communication problems
- Missing weather observations
- Incomplete traffic records
- Data collection interruptions

The preprocessing stage therefore identifies missing values before model training.

Depending on the variable and context, appropriate handling methods can include:

- Removing unusable observations
- Imputation
- Aggregation
- Using only records with sufficient information

The objective is to prevent missing or invalid data from negatively affecting model training.

---

# 🚨 Outlier Detection

Environmental datasets can contain extreme observations.

Some extreme values may represent genuine environmental events, while others may result from:

- Sensor errors
- Data-entry errors
- Communication problems
- Measurement anomalies

Outlier analysis is therefore performed before training.

Outliers should not automatically be removed because extreme environmental conditions may contain meaningful information.

---

# 🤖 Machine Learning Pipeline

After preprocessing and feature engineering, the final dataset is prepared for machine learning.

The general workflow is:

```text
Final Dataset
     │
     ▼
Select Features
     │
     ▼
Select Target
     │
     ▼
Train/Test Split
     │
     ├───────────────┐
     ▼               ▼
 Training Data     Testing Data
     │               │
     ▼               │
Model Training      │
     │               │
     ▼               │
Trained Model       │
     │               │
     └───────┬───────┘
             ▼
       Predictions
             │
             ▼
      Model Evaluation
```

---

# 🎯 Target Variable

The target variable represents the environmental quantity that the model is designed to predict.

The target is separated from the input features before model training.

Conceptually:

```text
Input Features (X)
        │
        ├── Weather
        ├── Traffic
        ├── Calendar
        ├── Historical environmental information
        └── Other engineered variables
        │
        ▼
Machine Learning Model
        │
        ▼
Predicted Target (y)
```

The exact target and feature selection are determined by the final modeling pipeline in the project.

---

# ✂️ Train-Test Split

The prepared dataset is divided into training and testing portions.

### Training Data

Used by the machine-learning algorithm to learn relationships between the input variables and target.

### Testing Data

Used to evaluate how well the trained model performs on previously unseen observations.

The separation is essential because evaluating a model using the same data on which it was trained can produce misleadingly high performance.

For time-dependent environmental data, the split should also be designed carefully to avoid information leakage from future observations into the training process.

---

# 🧠 Model Training

The machine-learning model is trained using the engineered dataset.

The model receives multiple environmental and contextual features and attempts to learn the relationship between these variables and the prediction target.

The general process is:

```text
X_train ─────────► Model ─────────► ŷ_train

X_test  ─────────► Trained Model ─► ŷ_test
```

Where:

- `X` = input features
- `y` = actual target
- `ŷ` = predicted target

---

# 📏 Model Evaluation

After training, predictions are generated using the test dataset.

The predictions are compared against the actual values.

Common regression evaluation metrics include:

### Mean Absolute Error (MAE)

Measures the average absolute difference between predicted and actual values.

```text
MAE = average(|Actual - Predicted|)
```

Lower MAE indicates better performance.

---

### Mean Squared Error (MSE)

Penalizes larger prediction errors more strongly.

```text
MSE = average((Actual - Predicted)²)
```

Lower MSE indicates better performance.

---

### Root Mean Squared Error (RMSE)

RMSE is the square root of MSE and is expressed in the same units as the target.

```text
RMSE = √MSE
```

Lower RMSE indicates better performance.

---

### R² Score

R² measures how much of the variation in the target is explained by the model.

```text
R² = 1 - (Residual Sum of Squares / Total Sum of Squares)
```

A value closer to 1 generally indicates better explanatory performance.

---

# 📊 Model Performance Analysis

Model performance should not be evaluated using a single metric alone.

The project evaluates the predictions by comparing:

```text
Actual Values
      vs
Predicted Values
```

Useful visualizations include:

### Actual vs Predicted Plot

A good regression model should produce predictions that are reasonably close to the diagonal relationship:

```text
Predicted
   │
   │       /
   │      /
   │     /
   │    /
   │   /
   └──────────── Actual
```

### Residual Analysis

Residuals are calculated as:

```text
Residual = Actual − Predicted
```

Residual analysis helps identify systematic prediction errors.

---

# 🔬 End-to-End Data Science Process

The complete project can be summarized as:

```text
1. Collect Raw Data
       ↓
2. Clean Individual Datasets
       ↓
3. Standardize Date/Time
       ↓
4. Standardize Locations
       ↓
5. Process CPCB Air Quality
       ↓
6. Process Copernicus Weather
       ↓
7. Process TomTom Traffic
       ↓
8. Process Public Holidays
       ↓
9. Integrate Datasets
       ↓
10. Create Calendar Features
       ↓
11. Create Engineered Features
       ↓
12. Perform EDA
       ↓
13. Select Features and Target
       ↓
14. Split Data
       ↓
15. Train Machine Learning Model
       ↓
16. Generate Predictions
       ↓
17. Evaluate Model
       ↓
18. Analyze Results
```

---

# 🗂️ Project Structure

The repository is organized around the different stages of the data-science workflow.

A conceptual structure is:

```text
Indian_weather_prediction/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── final/
│
├── notebooks/
│   ├── data preprocessing
│   ├── exploratory analysis
│   └── model training
│
├── scripts/
│   ├── data preparation
│   ├── feature engineering
│   └── model development
│
├── models/
│
├── visualizations/
│
├── README.md
│
└── requirements.txt
```

> The exact directory and filename structure may vary according to the files currently present in the repository.

---

# 🛠️ Technologies Used

The project uses the Python data-science ecosystem.

### Programming Language

- Python

### Data Processing

- Pandas
- NumPy

### Data Visualization

- Matplotlib
- Seaborn

### Machine Learning

- Scikit-learn

### Development Environment

- Jupyter Notebook / Python environment

### Data Sources

- Central Pollution Control Board (CPCB)
- Copernicus
- TomTom
- Indian public-holiday data

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/Vedant-Zade/Indian_weather_prediction.git
```

Move into the project directory:

```bash
cd Indian_weather_prediction
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Activate it on Linux/macOS:

```bash
source venv/bin/activate
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

The project can be reproduced by following the data-processing and modeling workflow in sequence.

### Step 1 — Prepare the source datasets

Place the raw datasets in their appropriate input directories.

### Step 2 — Run preprocessing

Clean and standardize the individual datasets.

### Step 3 — Integrate the datasets

Combine the prepared CPCB, Copernicus, TomTom, and public-holiday information using the standardized temporal and geographic fields.

### Step 4 — Generate engineered features

Create the calendar and other derived variables required by the model.

### Step 5 — Perform EDA

Generate the required statistical summaries and visualizations.

### Step 6 — Train the model

Use the prepared training data to fit the machine-learning model.

### Step 7 — Evaluate

Run predictions on the test dataset and calculate the selected evaluation metrics.

---

# 🔐 Data and Reproducibility Considerations

The project uses data originating from multiple external sources.

Because these sources may update their datasets or change their APIs/data formats over time, exact reproduction may require using the same historical data versions used during project development.

In particular:

- CPCB observations can change as additional observations become available.
- Copernicus datasets may be updated.
- Traffic information can vary over time.
- Public-holiday calendars depend on the selected year and source.

For reproducibility, the processed datasets used for model development should therefore be retained along with the code used to generate them.

---

# ⚠️ Limitations

Several limitations should be considered when interpreting the results.

### 1. Data Availability

Environmental datasets may contain missing observations or gaps.

### 2. Different Data Frequencies

The original sources may provide observations at different temporal frequencies. Combining them requires aggregation or alignment, which can result in some information loss.

### 3. Geographic Coverage

The quality of predictions depends on the geographic coverage and consistency of the underlying datasets.

### 4. External Factors

Air quality is influenced by many variables that may not be included in the dataset, such as:

- Industrial emissions
- Construction activity
- Crop burning
- Regional pollution transport
- Atmospheric boundary-layer conditions
- Local events
- Unusual weather events

### 5. Correlation Does Not Imply Causation

Relationships discovered during exploratory analysis do not necessarily establish that one variable directly causes another.

For example, a relationship between traffic and pollution does not by itself prove that traffic is the sole cause of a particular pollution event.

---

# 🚀 Future Improvements

The project can be extended in several directions.

## More Environmental Variables

Additional atmospheric variables can be incorporated to improve the representation of environmental conditions.

## More Detailed Traffic Information

Traffic data can be expanded to include:

- Road-level congestion
- Traffic speed
- Traffic volume
- Road type
- Hourly traffic patterns

## Time-Series Models

Since environmental observations are inherently temporal, specialized models such as:

- ARIMA
- SARIMA
- Prophet
- LSTM
- GRU
- Temporal Transformer models

could be investigated.

## Hyperparameter Optimization

Model performance can potentially be improved using:

- Grid Search
- Random Search
- Bayesian optimization
- Cross-validation

## Feature Importance

Feature-importance methods can be used to determine which environmental, traffic, and calendar variables contribute most strongly to predictions.

Interpretability techniques such as SHAP can also be used to explain individual predictions.

## Real-Time Prediction

A future version could connect live weather, traffic, and air-quality APIs to create a real-time prediction system.

---

# 🔍 Key Contribution of the Project

The main contribution of this project is not simply training a machine-learning algorithm.

The important part is the **construction of a unified multi-source environmental dataset**.

The project brings together four different perspectives:

```text
                  ENVIRONMENT
                      │
       ┌──────────────┼──────────────┐
       │              │              │
       ▼              ▼              ▼
   AIR QUALITY      WEATHER        TRAFFIC
       │              │              │
       └──────────────┼──────────────┘
                      │
                      ▼
               HUMAN ACTIVITY
                      │
                      ▼
              CALENDAR EFFECTS
                      │
                      ▼
             FEATURE ENGINEERING
                      │
                      ▼
              MACHINE LEARNING
                      │
                      ▼
                PREDICTION
```

Instead of treating air quality as an isolated phenomenon, the project attempts to represent it as the result of interacting **environmental, meteorological, traffic, and temporal factors**.

---

# 📚 Data Pipeline Summary

| Stage | Input | Output |
|---|---|---|
| Data Collection | Raw source datasets | Raw CPCB, weather, traffic and holiday data |
| Cleaning | Raw datasets | Clean datasets |
| Standardization | Clean datasets | Common formats |
| Temporal Alignment | Standardized datasets | Date-aligned observations |
| Geographic Alignment | Location information | Location-aligned observations |
| Integration | All processed sources | Combined dataset |
| Feature Engineering | Combined dataset | ML-ready features |
| EDA | ML-ready dataset | Statistical/visual insights |
| Data Splitting | Final dataset | Training/testing data |
| Model Training | Training data | Trained model |
| Prediction | Test/features | Predicted values |
| Evaluation | Predictions + actual values | Performance metrics |

---

# 📌 Conclusion

This project demonstrates an end-to-end machine-learning workflow for environmental prediction using heterogeneous real-world datasets.

The project begins with raw information collected from multiple independent sources and transforms it into a structured machine-learning dataset through extensive preprocessing, temporal alignment, geographic alignment, and feature engineering.

The four primary data sources provide complementary information:

- **CPCB** provides historical air-quality observations.
- **Copernicus** provides meteorological conditions.
- **TomTom** provides traffic-related information.
- **Public-holiday data** provides calendar and human-activity context.

The resulting dataset provides a richer representation of environmental conditions than any individual source could provide on its own.

The machine-learning stage then uses this integrated information to learn relationships between the engineered input variables and the prediction target.

Overall, the project demonstrates how **data integration and feature engineering are as important as model selection** when building a machine-learning solution from real-world environmental data.

---

# 👨‍💻 Authors

**Vedant Zade**

GitHub: [Vedant-Zade](https://github.com/Vedant-Zade)

---

# 📜 License

This project is intended for educational and research purposes.

Please refer to the licenses and usage terms of the individual external data sources before redistributing their original datasets.
