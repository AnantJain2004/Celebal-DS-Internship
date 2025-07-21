import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import load_iris, load_wine
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="ML Model Training Hub",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #333;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">🤖 ML Model Training Hub</h1>', unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox(
    "Choose the App Mode",
    ["Home", "Dataset Explorer", "Model Training", "Model Prediction", "Model Comparison"]
)

# Initialize session state
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'datasets' not in st.session_state:
    st.session_state.datasets = {}

@st.cache_data
def load_sample_datasets():
    """Load real datasets from various sources"""
    datasets = {}
    
    # 1. California Housing Dataset (Real dataset from sklearn)
    try:
        from sklearn.datasets import fetch_california_housing
        california = fetch_california_housing()
        california_df = pd.DataFrame(california.data, columns=california.feature_names)
        california_df['MedianHouseValue'] = california.target
        datasets['California Housing'] = california_df
        
    except Exception as e:
        st.error(f"Could not load California Housing dataset: {e}")
    
    # 2. Load Iris dataset (Real botanical dataset)
    try:
        iris = load_iris()
        iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
        # Map target numbers to actual species names
        species_names = ['Setosa', 'Versicolor', 'Virginica']
        iris_df['Species'] = [species_names[i] for i in iris.target]
        datasets['Iris Flowers'] = iris_df
    except Exception as e:
        st.error(f"Could not load Iris dataset: {e}")
    
    # 3. Wine Quality Dataset (Real wine quality data)
    try:
        wine = load_wine()
        wine_df = pd.DataFrame(wine.data, columns=wine.feature_names)
        # Map target numbers to wine class names
        class_names = ['Class_0', 'Class_1', 'Class_2']
        wine_df['Wine_Class'] = [class_names[i] for i in wine.target]
        datasets['Wine Quality'] = wine_df
    except Exception as e:
        st.error(f"Could not load Wine dataset: {e}")
    
    # 4. Diabetes Dataset (Real medical dataset)
    try:
        from sklearn.datasets import load_diabetes
        diabetes = load_diabetes()
        diabetes_df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
        diabetes_df['Disease_Progression'] = diabetes.target
        datasets['Diabetes Progression'] = diabetes_df
    except Exception as e:
        st.error(f"Could not load Diabetes dataset: {e}")
    
    # 5. Breast Cancer Dataset (Real medical dataset)
    try:
        from sklearn.datasets import load_breast_cancer
        cancer = load_breast_cancer()
        cancer_df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
        # Map target to meaningful labels
        cancer_df['Diagnosis'] = ['Malignant' if x == 0 else 'Benign' for x in cancer.target]
        datasets['Breast Cancer'] = cancer_df
    except Exception as e:
        st.error(f"Could not load Breast Cancer dataset: {e}")
    
    # 6. Load Titanic Dataset from seaborn (Real historical dataset)
    try:
        titanic_df = sns.load_dataset('titanic')
        # Clean the dataset
        titanic_df = titanic_df.dropna(subset=['age', 'fare'])
        
        # Convert categorical variables to numerical
        titanic_df['sex_encoded'] = titanic_df['sex'].map({'male': 1, 'female': 0})
        titanic_df['embarked_encoded'] = titanic_df['embarked'].map({'S': 0, 'C': 1, 'Q': 2})
        titanic_df['class_encoded'] = titanic_df['class'].map({'First': 1, 'Second': 2, 'Third': 3})
        
        # Select relevant columns
        titanic_clean = titanic_df[['survived', 'pclass', 'sex_encoded', 'age', 'sibsp', 
                                   'parch', 'fare', 'embarked_encoded', 'class_encoded']].dropna()
        titanic_clean['Survived'] = titanic_clean['survived'].map({0: 'No', 1: 'Yes'})
        
        datasets['Titanic Survival'] = titanic_clean
    except Exception as e:
        st.error(f"Could not load Titanic dataset: {e}")
    
    # 7. Tips Dataset (Real restaurant data)
    try:
        tips_df = sns.load_dataset('tips')
        # Encode categorical variables
        tips_df['sex_encoded'] = tips_df['sex'].map({'Male': 1, 'Female': 0})
        tips_df['smoker_encoded'] = tips_df['smoker'].map({'Yes': 1, 'No': 0})
        tips_df['day_encoded'] = tips_df['day'].map({'Thur': 0, 'Fri': 1, 'Sat': 2, 'Sun': 3})
        tips_df['time_encoded'] = tips_df['time'].map({'Lunch': 0, 'Dinner': 1})
        
        datasets['Restaurant Tips'] = tips_df
    except Exception as e:
        st.error(f"Could not load Tips dataset: {e}")
    
    return datasets

def home_page():
    """Home page with app description"""
    st.markdown("""
    ## Welcome to the ML Model Training Hub! 🚀
    
    This application demonstrates how to deploy machine learning models using Streamlit. 
    You can explore datasets, train models, make predictions, and visualize results all in one place.
    
    ### Features:
    - 📊 **Dataset Explorer**: Explore and visualize your data
    - 🧠 **Model Training**: Train various ML algorithms
    - 🎯 **Model Prediction**: Make predictions with trained models
    - 📈 **Model Comparison**: Compare different models' performance
    
    ### Supported Models:
    - Linear Regression
    - Random Forest Regression
    - Logistic Regression
    - Random Forest Classification
    
    Use the sidebar to navigate between different sections!
    """)
    
    # Display sample metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Models Available", "4")
    with col2:
        st.metric("Real Datasets", "7")
    with col3:
        st.metric("Visualizations", "Multiple")
    with col4:
        st.metric("Accuracy", "Up to 98%")
    
    # Dataset Information
    st.subheader("📊 Available Real Datasets")
    
    dataset_info = """
    | Dataset | Type | Description | Samples |
    |---------|------|-------------|---------|
    | **California Housing** | Regression | Real estate prices in California districts | 20,640 |
    | **Iris Flowers** | Classification | Botanical measurements of iris species | 150 |
    | **Wine Quality** | Classification | Chemical analysis of wines | 178 |
    | **Diabetes Progression** | Regression | Medical data for diabetes progression | 442 |
    | **Breast Cancer** | Classification | Medical diagnosis data | 569 |
    | **Titanic Survival** | Classification | Historical passenger survival data | ~800 |
    | **Restaurant Tips** | Regression | Real restaurant tip data | 244 |
    """
    
    st.markdown(dataset_info)

def dataset_explorer():
    """Dataset exploration page"""
    st.markdown('<h2 class="sub-header">📊 Dataset Explorer</h2>', unsafe_allow_html=True)
    
    # Load datasets
    datasets = load_sample_datasets()
    
    # Dataset selection
    dataset_choice = st.selectbox("Choose a dataset:", list(datasets.keys()))
    
    if dataset_choice:
        df = datasets[dataset_choice]
        st.session_state.datasets[dataset_choice] = df
        
        # Dataset overview
        st.subheader("Dataset Overview")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Shape:** {df.shape}")
            st.write(f"**Columns:** {len(df.columns)}")
        
        with col2:
            st.write(f"**Missing Values:** {df.isnull().sum().sum()}")
            st.write(f"**Duplicates:** {df.duplicated().sum()}")
        
        # Display data
        st.subheader("Dataset Preview")
        st.dataframe(df.head(10))
        
        # Statistical summary
        st.subheader("Statistical Summary")
        st.dataframe(df.describe())
        
        # Correlation heatmap
        st.subheader("Correlation Heatmap")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
        
        # Distribution plots
        st.subheader("Feature Distributions")
        if len(numeric_cols) > 0:
            selected_col = st.selectbox("Select column for distribution plot:", numeric_cols)
            
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots()
                ax.hist(df[selected_col], bins=30, alpha=0.7)
                ax.set_title(f'Distribution of {selected_col}')
                st.pyplot(fig)
            
            with col2:
                fig = px.box(df, y=selected_col, title=f'Box Plot of {selected_col}')
                st.plotly_chart(fig, use_container_width=True)

def model_training():
    """Model training page"""
    st.markdown('<h2 class="sub-header">🧠 Model Training</h2>', unsafe_allow_html=True)
    
    # Load datasets
    datasets = load_sample_datasets()
    
    if not datasets:
        st.error("No datasets available. Please check the Dataset Explorer first.")
        return
    
    # Dataset selection
    dataset_choice = st.selectbox("Choose a dataset for training:", list(datasets.keys()))
    
    if dataset_choice:
        df = datasets[dataset_choice]
        
        # Feature and target selection
        st.subheader("Select Features and Target")
        
        columns = df.columns.tolist()
        
        # Target selection
        target_col = st.selectbox("Select target column:", columns)
        
        # Feature selection
        feature_cols = st.multiselect(
            "Select feature columns:",
            [col for col in columns if col != target_col],
            default=[col for col in columns if col != target_col][:5]  # Default first 5
        )
        
        if feature_cols and target_col:
            X = df[feature_cols]
            y = df[target_col]
            
            # Determine problem type
            if len(y.unique()) <= 10 and y.dtype in ['object', 'int64']:
                problem_type = "classification"
                st.info("🎯 Detected: Classification Problem")
                model_options = ["Logistic Regression", "Random Forest Classifier"]
            else:
                problem_type = "regression"
                st.info("📈 Detected: Regression Problem")
                model_options = ["Linear Regression", "Random Forest Regressor"]
            
            # Model selection
            model_choice = st.selectbox("Choose a model:", model_options)
            
            # Train-test split
            test_size = st.slider("Test size ratio:", 0.1, 0.5, 0.2)
            
            if st.button("Train Model"):
                with st.spinner("Training model..."):
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42
                    )
                    
                    # Handle categorical variables for classification
                    if y.dtype == 'object' or problem_type == "classification":
                        le = LabelEncoder()
                        if y.dtype == 'object':
                            y_train_encoded = le.fit_transform(y_train)
                            y_test_encoded = le.transform(y_test)
                        else:
                            y_train_encoded = y_train
                            y_test_encoded = y_test
                    else:
                        y_train_encoded = y_train
                        y_test_encoded = y_test
                    
                    # Train model
                    if model_choice == "Linear Regression":
                        model = LinearRegression()
                    elif model_choice == "Random Forest Regressor":
                        model = RandomForestRegressor(n_estimators=100, random_state=42)
                    elif model_choice == "Logistic Regression":
                        model = LogisticRegression(random_state=42)
                    elif model_choice == "Random Forest Classifier":
                        model = RandomForestClassifier(n_estimators=100, random_state=42)
                    
                    model.fit(X_train, y_train_encoded)
                    
                    # Predictions
                    y_pred = model.predict(X_test)
                    
                    # Store model
                    model_key = f"{dataset_choice}_{model_choice}"
                    st.session_state.models[model_key] = {
                        'model': model,
                        'features': feature_cols,
                        'target': target_col,
                        'problem_type': problem_type,
                        'X_test': X_test,
                        'y_test': y_test_encoded if problem_type == "classification" else y_test,
                        'y_pred': y_pred,
                        'dataset_name': dataset_choice
                    }
                    
                    # Display results
                    st.success("✅ Model trained successfully!")
                    
                    # Metrics
                    if problem_type == "regression":
                        mse = mean_squared_error(y_test_encoded, y_pred)
                        r2 = r2_score(y_test_encoded, y_pred)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Mean Squared Error", f"{mse:.4f}")
                        with col2:
                            st.metric("R² Score", f"{r2:.4f}")
                        
                        # Prediction vs Actual plot
                        fig = px.scatter(
                            x=y_test_encoded, y=y_pred,
                            title="Actual vs Predicted Values",
                            labels={'x': 'Actual', 'y': 'Predicted'}
                        )
                        fig.add_trace(go.Scatter(
                            x=[y_test_encoded.min(), y_test_encoded.max()],
                            y=[y_test_encoded.min(), y_test_encoded.max()],
                            mode='lines',
                            name='Perfect Prediction',
                            line=dict(color='red', dash='dash')
                        ))
                        st.plotly_chart(fig, use_container_width=True)
                        
                    else:  # classification
                        accuracy = accuracy_score(y_test_encoded, y_pred)
                        
                        st.metric("Accuracy", f"{accuracy:.4f}")
                        
                        # Classification report
                        st.subheader("Classification Report")
                        report = classification_report(y_test_encoded, y_pred, output_dict=True)
                        report_df = pd.DataFrame(report).transpose()
                        st.dataframe(report_df)

def model_prediction():
    """Model prediction page"""
    st.markdown('<h2 class="sub-header">🎯 Model Prediction</h2>', unsafe_allow_html=True)
    
    if not st.session_state.models:
        st.warning("No trained models available. Please train a model first.")
        return
    
    # Model selection
    model_choice = st.selectbox("Choose a trained model:", list(st.session_state.models.keys()))
    
    if model_choice:
        model_info = st.session_state.models[model_choice]
        model = model_info['model']
        features = model_info['features']
        
        st.subheader("Enter Input Values")
        
        # Create input fields for each feature
        input_data = {}
        cols = st.columns(2)
        
        for i, feature in enumerate(features):
            with cols[i % 2]:
                input_data[feature] = st.number_input(
                    f"{feature}:",
                    value=0.0,
                    key=f"input_{feature}"
                )
        
        if st.button("Make Prediction"):
            # Create input dataframe
            input_df = pd.DataFrame([input_data])
            
            # Make prediction
            prediction = model.predict(input_df)[0]
            
            # Display prediction
            st.markdown(f"""
            <div class="success-box">
                <h3>Prediction Result: {prediction:.4f}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Show feature importance (if available)
            if hasattr(model, 'feature_importances_'):
                st.subheader("Feature Importance")
                importance_df = pd.DataFrame({
                    'Feature': features,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig = px.bar(
                    importance_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title='Feature Importance'
                )
                st.plotly_chart(fig, use_container_width=True)

def model_comparison():
    """Model comparison page"""
    st.markdown('<h2 class="sub-header">📈 Model Comparison</h2>', unsafe_allow_html=True)
    
    if len(st.session_state.models) < 2:
        st.warning("Need at least 2 trained models for comparison.")
        return
    
    # Select models to compare
    models_to_compare = st.multiselect(
        "Select models to compare:",
        list(st.session_state.models.keys()),
        default=list(st.session_state.models.keys())[:2]
    )
    
    if len(models_to_compare) >= 2:
        comparison_data = []
        
        for model_key in models_to_compare:
            model_info = st.session_state.models[model_key]
            
            if model_info['problem_type'] == 'regression':
                mse = mean_squared_error(model_info['y_test'], model_info['y_pred'])
                r2 = r2_score(model_info['y_test'], model_info['y_pred'])
                
                comparison_data.append({
                    'Model': model_key,
                    'MSE': mse,
                    'R² Score': r2
                })
            else:  # classification
                accuracy = accuracy_score(model_info['y_test'], model_info['y_pred'])
                
                comparison_data.append({
                    'Model': model_key,
                    'Accuracy': accuracy
                })
        
        # Display comparison
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df)
        
        # Visualization
        if 'Accuracy' in comparison_df.columns:
            fig = px.bar(
                comparison_df,
                x='Model',
                y='Accuracy',
                title='Model Accuracy Comparison'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = px.bar(
                comparison_df,
                x='Model',
                y='R² Score',
                title='Model R² Score Comparison'
            )
            st.plotly_chart(fig, use_container_width=True)

# Main app logic
if app_mode == "Home":
    home_page()
elif app_mode == "Dataset Explorer":
    dataset_explorer()
elif app_mode == "Model Training":
    model_training()
elif app_mode == "Model Prediction":
    model_prediction()
elif app_mode == "Model Comparison":
    model_comparison()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🤖 ML Model Training Hub | Built with Streamlit | Data Science Internship Assignment</p>
</div>
""", unsafe_allow_html=True)