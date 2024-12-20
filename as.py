import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load combined dataset
@st.cache
def load_data():
    data = pd.read_csv("combined_air_quality_with_location.csv", parse_dates=["datetime"])
    return data

# Initialize the app
st.title("Air Quality Analysis and Prediction")
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a Page", ["Data Overview", "Exploratory Data Analysis", "Modeling & Prediction"])

# Load data
data = load_data()

if page == "Data Overview":
    st.header("Data Overview")
    st.write("### Dataset Sample")
    st.dataframe(data.head())

    st.write("### Dataset Information")
    st.write(f"**Shape:** {data.shape}")
    st.write(f"**Columns:** {list(data.columns)}")

    st.write("### Missing Values")
    missing_values = data.isnull().sum()
    missing_percentage = (missing_values / len(data)) * 100
    st.write(pd.DataFrame({"Missing Values": missing_values, "Missing Percentage (%)": missing_percentage}))

    st.write("### Summary Statistics")
    st.write(data.describe())

if page == "Exploratory Data Analysis":
    st.header("Exploratory Data Analysis")

    st.write("### Distribution of PM2.5 Levels")
    fig, ax = plt.subplots()
    sns.histplot(data["PM2.5"], kde=True, bins=30, ax=ax, color="blue")
    st.pyplot(fig)

    st.write("### Correlation Heatmap")
    numerical_cols = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
    corr_matrix = data[numerical_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

    st.write("### Seasonal Variation in PM2.5")
    fig, ax = plt.subplots()
    sns.boxplot(x=data["season"], y=data["PM2.5"], palette="Set2", ax=ax)
    st.pyplot(fig)

if page == "Modeling & Prediction":
    st.header("Modeling & Prediction")

    st.write("### Data Preparation")
    X = data.drop(columns=["PM2.5", "datetime", "season", "site_name"])  # Features
    y = data["PM2.5"]  # Target

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocessing
    numerical_cols = ['PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
    categorical_cols = ['station']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    st.write(f"**Training Set Size:** {X_train.shape}")
    st.write(f"**Test Set Size:** {X_test.shape}")

    st.write("### Model Training")
    model_choice = st.selectbox("Choose a Model", ["Linear Regression", "Random Forest"])

    if model_choice == "Linear Regression":
        model = LinearRegression()
    elif model_choice == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)

    model.fit(X_train_processed, y_train)
    y_pred = model.predict(X_test_processed)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
    st.write(f"**R² Score:** {r2:.2f}")

    st.write("### Actual vs Predicted Values")
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.5, color="blue")
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color="red")
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    st.pyplot(fig)
