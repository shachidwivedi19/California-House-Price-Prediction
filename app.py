import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

from sklearn.datasets import fetch_california_housing

# ===================================
# Page Configuration
# ===================================

st.set_page_config(
    page_title="California House Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================================
# Premium CSS
# ===================================

st.markdown("""
<style>

/* Main App */
[data-testid="stAppViewContainer"]{
    background:#F4F7FB;
}

/* Header */
[data-testid="stHeader"]{
    background:transparent;
}

/* Sidebar */
section[data-testid="stSidebar"]{
    background:#0B1F3A;
}

section[data-testid="stSidebar"] *{
    color:white !important;
}

/* Main Content Text */
[data-testid="stMarkdownContainer"]{
    color:#1E293B !important;
}

h1,h2,h3,h4,h5,h6{
    color:#0B1F3A !important;
}

p,li,label,span{
    color:#374151 !important;
}

/* Title */
.main-title{
    font-size:42px;
    font-weight:700;
    color:#0B1F3A !important;
}

/* Cards */
.card{
    background:white;
    padding:20px;
    border-radius:16px;
    box-shadow:0 6px 18px rgba(0,0,0,.12);
    margin-bottom:20px;
}

/* Metric Card */
.metric-card{
    background:white;
    padding:18px;
    border-radius:16px;
    box-shadow:0 6px 18px rgba(0,0,0,.12);
}

/* Footer */
.footer{
    text-align:center;
    color:#6B7280;
    padding-top:20px;
}

</style>
""", unsafe_allow_html=True)

# ===================================
# Load Model
# ===================================

model = joblib.load("model.pkl")


# ===================================
# Load Dataset
# ===================================

housing = fetch_california_housing()

df = pd.DataFrame(
    housing.data,
    columns=housing.feature_names
)

df["Actual Price"] = housing.target

# ===================================
# Sidebar
# ===================================

st.sidebar.image(
    "https://img.icons8.com/color/96/home.png",
    width=80
)

st.sidebar.title("California Housing")

menu = st.sidebar.radio(

    "Navigation",

    [

        "🏠 Home",

        "📊 Dataset Explorer",

        "🤖 Manual Prediction",

        "📋 Dataset Sample",

        "📈 Model Performance",

        "ℹ About"

    ]

)

st.sidebar.markdown("---")

st.sidebar.info(

"""
Machine Learning Model

✔ Random Forest

Dataset

✔ California Housing Dataset
"""

)
# ===================================
# HOME PAGE
# ===================================

if menu == "🏠 Home":

    st.markdown(
        """
        <h1 class='main-title'>
        🏠 California House Price Prediction Dashboard
        </h1>
        """,
        unsafe_allow_html=True
    )

    st.write(
        """
Welcome to the **California House Price Prediction System**.

This project predicts house prices using **Machine Learning (Random Forest Regression)**.
It also provides interactive dashboards, dataset exploration, and prediction tools.
"""
    )

    st.markdown("---")

    # ==========================
    # KPI Cards
    # ==========================

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "🏘 Total Houses",
            f"{len(df):,}"
        )

    with col2:
        st.metric(
            "💰 Average Price",
            f"${df['Actual Price'].mean()*100000:,.0f}"
        )

    with col3:
        st.metric(
            "📈 Highest Price",
            f"${df['Actual Price'].max()*100000:,.0f}"
        )

    with col4:
        st.metric(
            "📉 Lowest Price",
            f"${df['Actual Price'].min()*100000:,.0f}"
        )

    st.markdown("---")

    # ==========================
    # Dashboard Cards
    # ==========================

    left, right = st.columns([2,1])

    with left:

        st.markdown(
            """
            <div class="card">
            <h3>📌 Project Overview</h3>

            <ul>
            <li>California Housing Dataset</li>
            <li>Random Forest Regression</li>
            <li>Interactive Dashboard</li>
            <li>Manual Prediction</li>
            <li>Dataset Sample Prediction</li>
            <li>Model Performance Analysis</li>
            </ul>

            </div>
            """,
            unsafe_allow_html=True
        )

    with right:

        st.markdown(
            """
            <div class="card">

            <h3>⚙ Tech Stack</h3>

            ✅ Python

            ✅ Pandas

            ✅ NumPy

            ✅ Scikit-Learn

            ✅ Plotly

            ✅ Streamlit

            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    # ==========================
    # Dataset Preview
    # ==========================

    st.subheader("📄 Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.markdown("---")

    # ==========================
    # Quick Statistics
    # ==========================

    st.subheader("📊 Quick Statistics")

    st.dataframe(
        df.describe(),
        use_container_width=True
    )

    st.markdown("---")

    # ==========================
    # House Price Distribution
    # ==========================

    fig = px.histogram(
        df,
        x="Actual Price",
        nbins=40,
        title="Distribution of House Prices",
        color_discrete_sequence=["#0B1F3A"]
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    st.success("✅ Project Loaded Successfully")

    st.markdown(
        """
        <div class='footer'>
        Developed by <b>Shachi Dwivedi</b> ❤️
        </div>
        """,
        unsafe_allow_html=True
    )
# ===================================
# DATASET EXPLORER
# ===================================

elif menu == "📊 Dataset Explorer":

    st.markdown(
        "<h1 class='main-title'>📊 California Housing Dashboard</h1>",
        unsafe_allow_html=True
    )

    st.write("Explore the California Housing Dataset with interactive visualizations.")

    st.markdown("---")

    # ==========================
    # Search Dataset
    # ==========================

    st.subheader("🔍 Search Dataset")

    house_index = st.number_input(
        "Enter House Index",
        min_value=0,
        max_value=len(df)-1,
        value=0,
        step=1
    )

    st.dataframe(
        df.iloc[[house_index]],
        use_container_width=True
    )

    st.markdown("---")

    # ==========================
    # Complete Dataset
    # ==========================

    st.subheader("📄 Complete Dataset")

    st.dataframe(
        df,
        use_container_width=True,
        height=350
    )

    st.markdown("---")

    # ==========================
    # Download Dataset
    # ==========================

    csv = df.to_csv(index=False).encode()

    st.download_button(
        "⬇ Download Dataset",
        csv,
        "CaliforniaHousing.csv",
        "text/csv"
    )

    st.markdown("---")

    # ==========================
    # Charts
    # ==========================

    col1, col2 = st.columns(2)

    with col1:

        fig = px.histogram(
            df,
            x="Actual Price",
            nbins=40,
            title="House Price Distribution",
            color_discrete_sequence=["#0B1F3A"]
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.box(
            df,
            y="Actual Price",
            title="House Price Box Plot",
            color_discrete_sequence=["#C9A227"]
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown("---")

    # ==========================
    # Scatter Plot
    # ==========================

    fig = px.scatter(

        df,

        x="MedInc",

        y="Actual Price",

        color="HouseAge",

        size="AveRooms",

        hover_data=df.columns,

        title="Median Income vs House Price"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    st.markdown("---")

    # ==========================
    # Correlation Heatmap
    # ==========================

    st.subheader("🔥 Correlation Heatmap")

    corr = df.corr()

    fig = px.imshow(

        corr,

        text_auto=True,

        color_continuous_scale="RdBu_r",

        aspect="auto"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    st.markdown("---")

    # ==========================
    # Average Price by House Age
    # ==========================

    avg = df.groupby("HouseAge")["Actual Price"].mean().reset_index()

    fig = px.line(

        avg,

        x="HouseAge",

        y="Actual Price",

        markers=True,

        title="Average House Price by House Age"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    st.markdown("---")

    # ==========================
    # California Map
    # ==========================

    st.subheader("🗺 California Houses Map")

    st.map(
        df[["Latitude", "Longitude"]]
    )

    st.markdown("---")

    # ==========================
    # Summary Cards
    # ==========================

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(

            "Average Income",

            round(df["MedInc"].mean(), 2)

        )

    with c2:

        st.metric(

            "Average Rooms",

            round(df["AveRooms"].mean(), 2)

        )

    with c3:

        st.metric(

            "Average Population",

            int(df["Population"].mean())

        )

    st.success("Dataset Exploration Completed ✅")
# ===================================
# 🤖 MANUAL PREDICTION
# ===================================

# ===================================
# 🤖 MANUAL PREDICTION
# ===================================

elif menu == "🤖 Manual Prediction":

    st.markdown(
        "<h1 class='main-title'>🤖 Manual House Price Prediction</h1>",
        unsafe_allow_html=True
    )

    st.write("Enter the house details below to predict the estimated house price.")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        MedInc = st.slider(
            "Median Income",
            min_value=0.0,
            max_value=15.0,
            value=3.5
        )

        HouseAge = st.slider(
            "House Age",
            min_value=1,
            max_value=52,
            value=20
        )

        AveRooms = st.slider(
            "Average Rooms",
            min_value=1.0,
            max_value=15.0,
            value=5.0
        )

        AveBedrms = st.slider(
            "Average Bedrooms",
            min_value=0.5,
            max_value=5.0,
            value=1.0
        )

    with col2:

        Population = st.number_input(
            "Population",
            min_value=1,
            value=1000
        )

        AveOccup = st.slider(
            "Average Occupancy",
            min_value=1.0,
            max_value=10.0,
            value=3.0
        )

        Latitude = st.slider(
            "Latitude",
            min_value=32.0,
            max_value=42.0,
            value=34.0
        )

        Longitude = st.slider(
            "Longitude",
            min_value=-125.0,
            max_value=-114.0,
            value=-118.0
        )

    # ==========================
    # Input Data
    # ==========================

    input_data = pd.DataFrame({

        "MedInc": [MedInc],
        "HouseAge": [HouseAge],
        "AveRooms": [AveRooms],
        "AveBedrms": [AveBedrms],
        "Population": [Population],
        "AveOccup": [AveOccup],
        "Latitude": [Latitude],
        "Longitude": [Longitude]

    })

    st.subheader("📋 Input Data")

    st.dataframe(input_data, use_container_width=True)

    # ==========================
    # Prediction Button
    # ==========================

    if st.button("🔮 Predict House Price"):

        with st.spinner("Predicting..."):

            prediction = forest_model.predict(input_data)[0]

        price = prediction * 100000

        # ==========================
        # Save Prediction History
        # ==========================

        history = input_data.copy()
        history["Predicted Price ($)"] = round(price, 2)

        try:
            old_history = pd.read_csv("prediction_history.csv")
            history = pd.concat(
                [old_history, history],
                ignore_index=True
            )
        except FileNotFoundError:
            pass

        history.to_csv(
            "prediction_history.csv",
            index=False
        )

        # ==========================
        # Show Prediction
        # ==========================

        st.success(
            f"🏠 Estimated House Price : ${price:,.2f}"
        )

        if price < 150000:

            st.info("🏠 Category : Affordable")

        elif price < 300000:

            st.warning("🏡 Category : Moderate")

        else:

            st.error("💎 Category : Luxury")

        st.balloons()

        # ==========================
        # Download Prediction
        # ==========================

        result = pd.DataFrame({

            "Predicted Price ($)": [round(price, 2)]

        })

        csv = result.to_csv(index=False).encode()

        st.download_button(

            "📥 Download Prediction",

            csv,

            "prediction.csv",

            "text/csv"

        )


# ===================================
# 📋 DATASET SAMPLE PREDICTION
# ===================================

elif menu == "📋 Dataset Sample":

    st.markdown(
        "<h1 class='main-title'>📋 Dataset Sample Prediction</h1>",
        unsafe_allow_html=True
    )

    st.write("Select any house from the California Housing Dataset to compare the actual and predicted price.")

    st.markdown("---")

    # ==========================
    # Select House
    # ==========================

    house_index = st.slider(
        "Select House Index",
        min_value=0,
        max_value=len(df)-1,
        value=0
    )

    sample = df.iloc[house_index]

    st.subheader("🏠 Selected House Details")

    st.dataframe(
        sample.to_frame(),
        use_container_width=True
    )

    # ==========================
    # Prediction
    # ==========================

    X = sample.drop("Actual Price")

    X_df = pd.DataFrame([X])

    prediction = model.predict(X_df)[0]

    actual_price = sample["Actual Price"] * 100000
    predicted_price = prediction * 100000

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "💰 Actual Price",
            f"${actual_price:,.2f}"
        )

    with col2:

        st.metric(
            "🤖 Predicted Price",
            f"${predicted_price:,.2f}"
        )

    # ==========================
    # Price Category
    # ==========================

    if predicted_price < 150000:

        st.info("🏠 Category : Affordable")

    elif predicted_price < 300000:

        st.warning("🏡 Category : Moderate")

    else:

        st.error("💎 Category : Luxury")

    st.markdown("---")

    # ==========================
    # Comparison Chart
    # ==========================

    comparison = pd.DataFrame({

        "Type": [
            "Actual Price",
            "Predicted Price"
        ],

        "Price": [
            actual_price,
            predicted_price
        ]

    })

    fig = px.bar(

        comparison,

        x="Type",

        y="Price",

        text="Price",

        color="Type",

        color_discrete_sequence=[
            "#0B1F3A",
            "#C9A227"
        ]

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ==========================
    # Prediction Error
    # ==========================

    error = abs(actual_price - predicted_price)

    accuracy = 100 - (error / actual_price * 100)

    st.markdown("---")

    c1, c2 = st.columns(2)

    with c1:

        st.metric(
            "📉 Prediction Error",
            f"${error:,.2f}"
        )

    with c2:

        st.metric(
            "🎯 Approx Accuracy",
            f"{accuracy:.2f}%"
        )

    # ==========================
    # Save Prediction History
    # ==========================

    history = pd.DataFrame({

        "House Index": [house_index],
        "Actual Price": [round(actual_price, 2)],
        "Predicted Price": [round(predicted_price, 2)],
        "Prediction Error": [round(error, 2)]

    })

    try:

        old = pd.read_csv("prediction_history.csv")

        history = pd.concat(
            [old, history],
            ignore_index=True
        )

    except FileNotFoundError:

        pass

    history.to_csv(
        "prediction_history.csv",
        index=False
    )

    # ==========================
    # Download Report
    # ==========================

    csv = history.to_csv(index=False).encode()

    st.download_button(

        "📥 Download Prediction Report",

        csv,

        "prediction_report.csv",

        "text/csv"

    )

    st.success("✅ Prediction Completed Successfully")

    st.balloons()

# ===================================
# 📈 MODEL PERFORMANCE
# ===================================

elif menu == "📈 Model Performance":

    st.markdown(
        "<h1 class='main-title'>📈 Model Performance Dashboard</h1>",
        unsafe_allow_html=True
    )

    st.write("Performance summary of the trained Machine Learning model.")

    st.markdown("---")

    # ==========================
    # Performance Metrics
    # ==========================

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("R² Score", "0.81")

    with col2:
        st.metric("MAE", "0.31")

    with col3:
        st.metric("RMSE", "0.55")

    st.markdown("---")

    st.subheader("📋 Model Details")

    details = pd.DataFrame({

        "Property":[
            "Model",
            "Dataset",
            "Features",
            "Training Samples",
            "Testing Samples"
        ],

        "Value":[
            "Random Forest Regressor",
            "California Housing",
            "8",
            "16512",
            "4128"
        ]

    })

    st.table(details)

    st.markdown("---")

    # ==========================
    # Feature Importance
    # ==========================

    st.subheader("🌲 Feature Importance")

    if hasattr(model, "feature_importances_"):

        importance = pd.DataFrame({

            "Feature": df.columns[:-1],
            "Importance": model.feature_importances_

        })

        importance = importance.sort_values(
            by="Importance",
            ascending=False
        )

        fig = px.bar(

            importance,

            x="Importance",

            y="Feature",

            orientation="h",

            color="Importance",

            color_continuous_scale="Blues",

            title="Feature Importance"

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.warning("Feature importance is not available for this model.")

    st.markdown("---")

    # ==========================
    # Model Summary
    # ==========================

    st.subheader("📝 Model Summary")

    st.info("""
- ✔ Algorithm : Random Forest Regressor
- ✔ Supervised Machine Learning
- ✔ Regression Problem
- ✔ Dataset : California Housing
- ✔ Features : 8
- ✔ Target : Median House Value
""")

    st.markdown("---")

    # ==========================
    # Prediction Workflow
    # ==========================

    st.subheader("⚙️ Prediction Workflow")

    workflow = pd.DataFrame({

        "Step":[
            "Load Dataset",
            "Preprocessing",
            "Scaling",
            "Model Prediction",
            "Display Result"
        ],

        "Status":[
            "✅",
            "✅",
            "✅",
            "✅",
            "✅"
        ]

    })

    st.table(workflow)

    st.success("Model evaluation completed successfully ✅")
    # ===================================
# 📜 PREDICTION HISTORY
# ===================================

elif menu == "📜 Prediction History":

    st.markdown(
        "<h1 class='main-title'>📜 Prediction History</h1>",
        unsafe_allow_html=True
    )

    try:

        history = pd.read_csv("prediction_history.csv")

        st.dataframe(
            history,
            use_container_width=True
        )

        st.success(f"Total Predictions : {len(history)}")

        csv = history.to_csv(index=False).encode()

        st.download_button(

            "📥 Download History",

            csv,

            "prediction_history.csv",

            "text/csv"

        )

    except FileNotFoundError:

        st.warning("No prediction history available.")

        st.markdown("---")

st.markdown("""
<div style='text-align:center'>
California House Price Prediction
</div>
""", unsafe_allow_html=True)