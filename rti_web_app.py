import streamlit as st
from PIL import Image

# App Logo and Title
st.set_page_config(page_title="RTI Diagnosis Prediction", layout="wide")

st.subheader('',divider="rainbow")

# Custom logo and title in columns
logo = Image.open("logo.png")  # Replace with your logo image path
col1, col2 = st.columns([1, 5])
with col1:
    st.image(logo, width=100)
with col2:
    st.title("Respiratory Tract Infection Diagnosis Prediction")
# st.subheader('',divider="rainbow")

# Tab Navigation with Icons
tabs = st.tabs([
    "\U0001F3E0 Home",
    "\U0001F4BE Dataset",
    "\U0001F527 Methodology",
    "\U0001F4CA Exploratory Data Analysis",
    "\U0001F50D Feature Selection",
    "\U0001F4BB Modeling",
    "\U0001F4C9 Model Evaluation",
    "\U0001F501 Run Model"
])

# Tab Content
with tabs[0]:
    # Embed a YouTube video representing respiratory tract infection
    st.components.v1.html(
    """
    <div style="display: flex; justify-content: center;">
        <iframe width='560' height='315' src='https://www.youtube.com/embed/L1x6LoL7pTw?si=bp_QPRlKMzY-prFs&start=6' 
        title='YouTube video player' frameborder='0' allow='accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; 
        picture-in-picture; web-share' referrerpolicy='strict-origin-when-cross-origin' allowfullscreen></iframe>
    </div>
    """,
    height=315
)

    # Add a subheader for problem statement
    st.subheader("Problem Statement")
    st.write("""
        **Identification of Respiratory Tract Infection**

        Accurately diagnosing the presence of a respiratory tract infection using 
        patient data and relevant clinical features. This stage aims to enhance early detection and streamline 
        the diagnostic process, ensuring timely intervention.

        The sole purpose of this project is to expedite the identification of immune medicine tailored to the patient's needs. 
        **By leveraging predictive analytics and machine learning, this project seeks to bypass the time-consuming 
        process of laboratory testing from collected samples. Instead, it enables clinicians to determine suitable 
        immune medications promptly.** This approach not only accelerates treatment but also reduces the burden 
        on diagnostic laboratories, ensuring that critical cases receive attention swiftly. 
    """)

with tabs[1]:
    # Main tab for Dataset Overview
    # st.header("Dataset Overview")
    
    # Nested tabs for Overview and Data
    sub_tabs = st.tabs(["Data Description", " Raw Data"])
    
    with sub_tabs[0]:
    # Content for Overview of the Data
        st.subheader("Data Description")
        st.markdown("""
    The dataset is organized into several distinct sections, each containing features critical to the analysis and prediction of respiratory tract infections (RTIs).
    Below is a detailed breakdown of the sections and their features, including notes on data availability.

    ### 1. General Information
    This section contains demographic and clinical identifiers.
    
    **Available Features:**
    - **S.No**: Serial number for record identification.
    - **OrderDate**: Date of the clinical order or sample collection.
    - **Reg No**: Patient registration number.
    - **AGE**: Patient's age.
    - **GENDER**: Gender of the patient.
    - **Ward**: Ward of admission (if applicable).
    - **Dept**: Department managing the case.
    - **SAMPLE**: Type of sample collected for investigation.

    ### 2. Personal Data
    This section highlights personal and environmental risk factors.
    
    **Available Features:**
    - **Smoking**: Smoking status of the patient (Yes or No).
    - **Asthma/COPD**: Presence of asthma or chronic obstructive pulmonary disease (Yes or No).
    - **Marital Status**: Marital status of the patient (Yes or No).

    ### 3. Presenting Complaints
    Captures symptoms reported during the clinical visit.
    
    **Available Features:**
    - **Acute Cough Duration**: Duration of acute cough (Single value throughout all records).
    - **Sputum**: Sputum production details (Data available only if the sample is sputum).
    - **Fever**: Presence and severity of fever (Single value throughout all records).
    - **Chills + Rigors**: Presence of chills and rigors (Single value throughout all records).
    - **Pleuritic Chest Pain**: Chest pain associated with breathing (Single value throughout all records).
    - **Difficulty Breathing**: Breathing issues or shortness of breath.
    - **Headache**: Headache occurrence (Single value throughout all records).
    - **H/O Generalized Weakness**: History of generalized weakness (Single value throughout all records).
    - **Mental Status**: Mental state of the patient during presentation (Single value throughout all records).

    ### 4. Comorbidities
    Details pre-existing medical conditions that could influence disease severity.
    
    **Available Features:**
    - **Comorbidities**: Summary of all comorbidities.
    - **Myocardial Infarction**: Heart attack history (Single value throughout all records).
    - **Congestive Heart Failure**: Presence of heart failure (Single value throughout all records).
    - **Peripheral Vascular Disease**: Circulatory system disorders (Single value throughout all records).
    - **Cerebrovascular Disease**: Stroke or related conditions (Single value throughout all records).
    - **Chronic Pulmonary Disease/Asthma**: Chronic lung conditions or asthma (Single value throughout all records).
    - **Connective Tissue Disease**: Disorders like lupus or rheumatoid arthritis (Single value throughout all records).
    - **Peptic Ulcer Disease**: History of peptic ulcers (Single value throughout all records).
    - **Moderate or Severe Liver Disease**: Liver dysfunction severity (Single value throughout all records).
    - **Moderate or Severe Renal Disease**: Kidney dysfunction severity (Single value throughout all records).
    - **Diabetes Without End-Organ Damage**: Controlled diabetes without complications (Single value throughout all records).
    - **Diabetes With End-Organ Damage**: Complicated diabetes (Single value throughout all records).
    - **AIDS**: Presence of acquired immunodeficiency syndrome (Single value throughout all records).
    - **Recent Immunosuppressive Therapy/Chemotherapy**: History of treatments affecting immunity (Single value throughout all records).
    - **Lung Malignancy**: Lung cancer presence (Single value throughout all records).

    ### 5. Red Flag Signs
    Indicators of severe or urgent conditions requiring immediate attention.
    
    **Available Features:**
    - **Chest Pain/Difficulty in or Rapid Breathing**: Severe respiratory distress or chest pain.

    ### Sections Without Data Availability
    Certain sections in the dataset have been defined but currently lack corresponding data. These sections are crucial for a more comprehensive analysis and may require further data collection efforts to enhance the study's robustness.
    
    **Sections without Data Availability:**
    - Personal Data (Few Features)
    - Presenting Complaints (Few Features)
    - Past Infection Data
    - Hospital Admission History
    - Drug History
    - Travel History
    - Clinical Parameters
    - Investigations
    - CURB-65 Score
    - Comorbidities (Few Features)
    - Red Flag Signs (Few Features)

    By integrating data into these sections, the overall analysis can achieve greater precision and provide a deeper understanding of the factors influencing respiratory tract infections.
    """)


    with sub_tabs[1]:
        # Content for Data Tab
        st.subheader(" Raw Data")
        
        # Upload the Excel file
        # uploaded_file = st.file_uploader("Upload your dataset file (Excel format)", type=["xlsx", "xls"])
        import pandas as pd
        uploaded_file = pd.read_excel("Autofinal.xlsx")
        
            
            # Load the dataset
        data = uploaded_file
            
            # Display a sample of the data
        st.write("### Sample Data (First 5 Rows)")
        st.dataframe(data.head())
            
            # Option to display the full dataset
        if st.checkbox("Show Full Dataset"):
            st.write("### Full Dataset")
            st.dataframe(data)
        
        
        # if uploaded_file is not None:
            
        # else:
        #     st.write("Please upload a dataset to view its contents.")


with tabs[2]:
    st.header("Methodology")
    
    # Placeholder for image
    # Create three columns
    left_col, center_col, right_col = st.columns(3)

# Place the image in the center column
    with center_col:
        st.image("methodologyflowchart.png",width = 1020)

    # image_placeholder = st.image("methodologyflowchart.png")  # Placeholder for adding an image later
    
    # Sub-tabs for specific sections
    sub_tabs = st.tabs(["Data Cleaning", "Feature Scaling & Encoding"])
    
    with sub_tabs[0]:
        st.subheader("Data Cleaning")
        st.write("""
        In the preprocessing phase, the dataset underwent a series of systematic steps to ensure data
        consistency, remove irrelevant information, and prepare the features for model training. The key
        actions included:
        
        1. **Correction of Spelling Mistakes and Abbreviations in the Diagnosis Column**
           - The Diagnosis column, which serves as the target variable, was thoroughly
             reviewed for spelling errors and non-standard abbreviations.
           - Identified errors were corrected to maintain consistency, and abbreviations were
             either expanded or standardized. This ensured that the model could effectively
             interpret and classify diagnoses without being affected by inconsistencies.
        """)
        # Create three columns
        left_col, center_col, right_col = st.columns(3)

# Place the image in the center column
        with center_col:
            st.image("spellmistakes_abs.png")

        # spell_mistakes_abs_image = st.image("spellmistakes_abs.png")
        st.write("""
        2. **Standardization of the AGE Column**
           - The AGE column contained values in varying formats, such as 75 Y (75 years)
             or 4Y 13D (4 years and 13 days).
           - These values were standardized by:
             - Converting entries like 75 Y to 75.
             - Simplifying entries like 4Y 13D to 4 by retaining only the year component.
             - Removing data containing only days, such as 13D, as they were deemed
               irrelevant.
        
        3. **Removal of Columns with Missing Data**
           - Columns with no data availability or consistently missing values were identified
             and removed from the dataset.
           - This step streamlined the dataset, reducing redundancy and ensuring that only
             meaningful features were retained for analysis.
        """)
    
    with sub_tabs[1]:
        st.subheader("Feature Selection and Encoding")
        st.write("""
        **Feature Scaling and Encoding**

        - **Numerical Columns:**
          - All numerical columns were scaled using the MinMax Scaler, which
            normalized the values to a range between 0 and 1. This scaling ensured
            uniformity across features, preventing any single feature from
            disproportionately influencing the model.
        """)
        # Create three columns
        left_col, center_col, right_col = st.columns(3)

# Place the image in the center column
        with center_col:
            st.image("minmaxscalar.png")

        # minmax_image = st.image("minmaxscalar.png")
        st.write("""
        - **Categorical Columns:**
          - Categorical data was encoded using Label Encoding, which assigned a
            unique numerical value to each category. This transformation enabled
            categorical data to be effectively utilized in machine learning algorithms.
        """)
        # Create three columns
        left_col, center_col, right_col = st.columns(3)

# Place the image in the center column
        with center_col:
            st.image("labelencoding.jpg")

        # label_encoding_image = st.image("labelencoding.jpg")

with tabs[3]:
    st.header("Exploratory Data Analysis")
    eda_sub_tabs = st.tabs(["Data Integrity Check", "Visualization"])

    with eda_sub_tabs[0]:
        # st.subheader("Data Integrity Check")
        st.write("""
        The Exploratory Data Analysis (EDA) phase focused on understanding the dataset's
structure, identifying patterns, and uncovering potential issues. This critical step provided
valuable insights into the data's distribution, relationships, and quality.
      
    """)
        # st.subheader("Assesing Unique Values , Duplicate and Missing Data")
        import pandas as pd
        data = pd.read_excel("Autofinal.xlsx")

        # Display column-wise unique values
        st.subheader("Column-wise Unique Values")
        unique_values = {col: data[col].nunique() for col in data.columns}
        unique_values_df = pd.DataFrame(list(unique_values.items()), columns=["Column Name", "Unique Value Count"])
        st.dataframe(unique_values_df)

        # Display column-wise duplicate values
        st.subheader("Record-wise Duplicate Values")
        duplicate_records = data[data.duplicated()]
        if not duplicate_records.empty:
            st.write(f"Number of duplicate records: {duplicate_records.shape[0]}")
            st.dataframe(duplicate_records)
        else:
            st.write("No duplicate records found.")

        # Display column-wise missing values
        st.subheader("Column-wise Count of Missing Values")
        missing_values = {col: data[col].isnull().sum() for col in data.columns}
        missing_values_df = pd.DataFrame(list(missing_values.items()), columns=["Column Name", "Missing Value Count"])
        st.dataframe(missing_values_df)
        
    with eda_sub_tabs[1]:
        st.subheader("Visualization")
        # st.write(""" Here visuals will be added...!! """)
        import pandas as pd
        import plotly.express as px
        import plotly.graph_objects as go
        
        data = pd.read_excel("Autofinal.xlsx")

        # Dropdown for selecting a column
        st.write("Select a column for visualization:")
        column_options = data.columns.tolist()

        # 1. Bar Plot for Categorical Columns
        st.subheader("Bar Plot")
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns
        if len(categorical_columns) > 0:
            selected_categorical = st.selectbox("Select a categorical Column",categorical_columns)
            if selected_categorical:
        # Count values and reset index for Plotly compatibility
                category_counts = data[selected_categorical].value_counts().reset_index()
                category_counts.columns = [selected_categorical, "count"]

        # Create a bar plot
                fig_bar = px.bar(
                    category_counts,
                    x=selected_categorical,
                    y="count",
                    text = "count",
                    labels={selected_categorical: "Category", "count": "Count"},
                    title=f"Distribution of {selected_categorical}"
                )
                fig_bar.update_traces(textposition = 'outside')
                st.plotly_chart(fig_bar)
        else:
            st.write("No categorical columns available for bar plot.")

        # 2. Histogram for Numerical Columns
        st.subheader("Histogram")
        numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns
        if len(numerical_columns) > 0:
            selected_numerical = st.selectbox("Select a numerical column", numerical_columns)
            if selected_numerical:
                fig_histogram = px.histogram(
                    data,
                    x=selected_numerical,
                    nbins=20,
                    title=f"Distribution of {selected_numerical}"
                )
                st.plotly_chart(fig_histogram)
        else:
            st.write("No numerical columns available for histogram.")

        # # 3. Scatter Plot
        # st.subheader("Scatter Plot")
        # if len(numerical_columns) > 1:
        #     scatter_x = st.selectbox("Select X-axis for scatter plot", numerical_columns, index=0)
        #     scatter_y = st.selectbox("Select Y-axis for scatter plot", numerical_columns, index=1)
        #     if scatter_x and scatter_y:
        #         fig_scatter = px.scatter(
        #             data,
        #             x=scatter_x,
        #             y=scatter_y,
        #             color=data[categorical_columns[0]] if len(categorical_columns) > 0 else None,
        #             title=f"Scatter Plot: {scatter_x} vs {scatter_y}",
        #             labels={scatter_x: scatter_x, scatter_y: scatter_y}
        #         )
        #         st.plotly_chart(fig_scatter)
        # # Scatter Plot: AGE vs GENDER
        # st.subheader("Scatter Plot: AGE vs GENDER")
        # if 'AGE' in numerical_columns and 'GENDER' in categorical_columns:
        #     fig_scatter = px.scatter(
        #         data,
        #         x="AGE",
        #         y="GENDER",
        #         color="GENDER",
        #         title="Scatter Plot: AGE vs GENDER",
        #         labels={"AGE": "Age", "GENDER": "Gender"},
        #         hover_data=data.columns,  # Add more details on hover
        #     )
        #     st.plotly_chart(fig_scatter)
        # else:
        #     st.write("Required columns 'AGE' and 'GENDER' are not available.")
        
        # 5. Pie Chart for Categorical Columns
        st.subheader("Pie Chart")
        if len(categorical_columns) > 0:
            pie_column = st.selectbox("Select a categorical column for pie chart", categorical_columns, index=0)
            if pie_column:
                fig_pie = px.pie(
                    data,
                    names=pie_column,
                    title=f"Pie Chart of {pie_column}",
                    hole=0.4
                )
                st.plotly_chart(fig_pie)

        # 6. Box Plot for Outlier Detection
        st.subheader("Box Plot")
        if len(numerical_columns) > 0:
            box_column = st.selectbox("Select a numerical column for box plot", numerical_columns, index=0)
            if box_column:
                fig_box = px.box(
                    data,
                    y=box_column,
                    title=f"Box Plot of {box_column}",
                    points="all"  # Show all points
                )
                st.plotly_chart(fig_box)


with tabs[4]:
    # st.header("Feature Selection")
    st.write("""
        `Recursive Feature Elimination (RFE)` was employed to identify 
        the top 10 features that most significantly impact the target variable, Diagnosis. This method
        systematically removes the least important features based on their contribution to the model's
        predictive power until the desired number of features is achieved.

        **Process:**
        1. **Model Selection for RFE:**
            - A Decision Tree Classifier was used as the estimator for RFE. Its ability to
            handle non-linear relationships and feature importance metrics makes it
            well-suited for this task.
        2. **Feature Ranking:**
            - RFE iteratively trained the model by removing the weakest features at each step,
            based on their importance scores.
            -  This process continued until only the top 10 features remained.
        3. **Results:**
            The training set was used for feature selection (Recursive Feature Elimination)
            and model building.
            - The testing set was used exclusively for validating the model’s accuracy and
              other performance metrics.
    """)

    # Create three columns
    # left_col, center_col, right_col = st.columns(3)

# Place the image in the center column
    # with center_col:
    st.image("rfe.jpg", width = 900)

    

    st.write("""**Significance of Feature Selection:**
- Reducing the dimensionality of the dataset helped focus on the most impactful
          predictors, improving model interpretability and reducing overfitting.
- These selected features were then used for training the predictive models, ensuring that
          the analysis remained efficient and meaningful.
    """)

# tabs = st.tabs(["Home", "Dataset", "Methodology", "Exploratory Data Analysis", "Feature Selection","Modeling", "Model Evaluation", "Run Model"])

with tabs[5]:
    st.header("Modeling")
    st.write("""
        A `Decision Tree Classifier` was implemented to predict the target variable, Diagnosis. 
        The model was trained on the selected top 10 features (determined via Recursive Feature Elimination) 
        from the training dataset and tested on the reserved test data.

        **Model Training:**
        1. **Model Selection:**
            - A Decision Tree Classifier was chosen for its interpretability and ability to 
              handle non-linear relationships.
        2. **Training Process:**
            - The model was trained using the training dataset consisting of the top 10 
              features and the target variable.
            - Hyperparameters such as `max_depth = 4` were tuned to ensure optimal 
              performance and avoid overfitting.
    """)


# tabs = st.tabs(["Home", "Data Integrity Check", "Visualization", "Feature Selection", "Modeling", "Model Evaluation"])

with tabs[6]:  # "Model Evaluation" tab
    # st.header("Model Evaluation")

    # Subtabs for "Performance Metrics" and "Results"
    sub_tabs = st.tabs(["Performance Metrics", "Results"])

    with sub_tabs[0]:  # Performance Metrics tab
        st.subheader("Model Evaluation: Performance Metrics")
        st.write("""
        The performance of the Decision Tree Classifier was assessed using several metrics to 
        understand its effectiveness in predicting the target variable, Diagnosis. Given the imbalanced 
        nature of the dataset, the F1-score was prioritized as the primary evaluation metric.

        **Evaluation Metrics Used:**
        1. **Accuracy:**
            - Measured the overall proportion of correctly predicted instances in the dataset.
            - While accuracy is an intuitive metric, it can be misleading for imbalanced 
              datasets as it may favor the majority class.
        2. **Precision:**
            - Defined as the proportion of true positive predictions out of all positive 
              predictions.
            - Precision highlights the model's ability to avoid false positives, which is important 
              for certain diagnoses.
        3. **Recall:**
            - Defined as the proportion of true positive predictions out of all actual positives.
            - Recall is critical for capturing all instances of a diagnosis, especially the minority 
              classes.
        4. **F1-Score:**
            - The harmonic mean of precision and recall.
            - Provides a single metric that balances false positives and false negatives, 
              making it ideal for imbalanced datasets.
        """)

    with sub_tabs[1]:  # Results tab
        st.subheader("Model Evaluation: Results")
        # st.write("Content for Results will be added here.")
        data = {
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
        "Max_depth = None": [0.56, 0.47, 0.56, 0.51],
        "Max_depth = 4": [0.65, 0.44, 0.65, 0.52]
    }
    # df_metrics = pd.DataFrame(data)
        df_metrics = pd.DataFrame(data)
        # data_editable = st.data_editor(df_metrics)
        

        left_col, right_col = st.columns(2)
        
        with left_col:
            st.subheader("Comparison")
            st.write(df_metrics)
        
        # Create two columns for the delta comparison
        with right_col:
            st.subheader("Max_depth = 4 Vs Max_depth=None")
            
            # First row of two metrics
            col1, col2 = st.columns(2)
            with col1:
                metric = df_metrics.iloc[0]
                delta = metric['Max_depth = 4'] - metric['Max_depth = None']
                st.metric(label=metric['Metric'], value=f"{metric['Max_depth = 4']:.2f}", delta=f"{delta:.2f}")
            
            with col2:
                metric = df_metrics.iloc[1]
                delta = metric['Max_depth = 4'] - metric['Max_depth = None']
                st.metric(label=metric['Metric'], value=f"{metric['Max_depth = 4']:.2f}", delta=f"{delta:.2f}")
            
            # Second row of two metrics
            col3, col4 = st.columns(2)
            with col3:
                metric = df_metrics.iloc[2]
                delta = metric['Max_depth = 4'] - metric['Max_depth = None']
                st.metric(label=metric['Metric'], value=f"{metric['Max_depth = 4']:.2f}", delta=f"{delta:.2f}")
            
            with col4:
                metric = df_metrics.iloc[3]
                delta = metric['Max_depth = 4'] - metric['Max_depth = None']
                st.metric(label=metric['Metric'], value=f"{metric['Max_depth = 4']:.2f}", delta=f"{delta:.2f}")
            
        st.write("""
            **Observations:**
            1. Setting `max_depth=4` improved the `accuracy` and `recall` of the model compared to `max_depth=None`, 
            **indicating better generalization and the ability to correctly identify true positives across classes**.
            2. However, the `precision` slightly decreased, possibly due to an increase in false positives.
            3. The `F1-Score` showed a marginal improvement, demonstrating a better balance between precision and recall 
            for `max_depth=4`.
            
            **Significance of F1-Score:**
            The `F1-score`, being a balance of precision and recall, provides a clearer understanding of the model's 
            performance on imbalanced datasets like this one. It ensures that both false positives and false negatives 
            are considered, preventing the model from being biased toward the majority class.
            """)


with tabs[7]:
    st.subheader("Run Model")
    
    # Subtabs for 'Upload Data' and 'Simulation'
    sub_tabs = st.tabs(["Upload Data", "Simulation"])

    # Subtab: Upload Data
    with sub_tabs[0]:
        st.subheader("Upload Data")

        # Columns for layout
        col1, col2 = st.columns(2)

        # Column 1: User Input Options
        with col1:
            # Dropdown for choosing input data
            input_choice = st.selectbox(
                "Choose the input data:",
                options=["Use the test data", "Upload the local dataset"]
            )

            # Show data uploader if "Upload the local dataset" is selected
            uploaded_file = None
            if input_choice == "Upload the local dataset":
                uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])

            # Toggle to use test data
            use_test_data_toggle = st.toggle("Use Test Data", value=(input_choice == "Use the test data"))

            # Slicer for selecting number of records
            num_records = st.slider(
                "Select number of records for the test data:",
                min_value=1,
                max_value=140,  # Replace with max records in your dataset
                value=2
            )

            # Run Model Button
            if st.button("Run Model"):
                # Load and process the data
                if use_test_data_toggle or input_choice == "Use the test data":
                    # Simulated test data
                    test_data = pd.DataFrame(
                        {
                            "Feature1": [0.1, 0.2, 0.3][:num_records],
                            "Feature2": [0.4, 0.5, 0.6][:num_records],
                            "Target": [1, 0, 1][:num_records]
                        }
                    )
                elif uploaded_file:
                    # Use uploaded dataset
                    test_data = pd.read_csv(uploaded_file).head(num_records)
                else:
                    st.error("No data selected!")
                    test_data = None

                # Process the selected data
                if test_data is not None:
                    # Simulated top features
                    top_features = ["Feature1", "Feature2"]

                    # Load the pretrained model
                    with open("pretrained_model.pkl", "rb") as model_file:
                        model = pickle.load(model_file)

                    # Filter data with top features
                    filtered_data = test_data[top_features]

                    # Generate predictions
                    predictions = model.predict(filtered_data)

                    # Simulated metrics
                    results = {
                        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
                        "Max_depth = None": [0.56, 0.47, 0.56, 0.51],
                        "Max_depth = 4": [0.65, 0.44, 0.65, 0.52]
                    }
                    df_metrics = pd.DataFrame(results)

                    # Column 2: Display Results
                    with col2:
                        st.subheader("Results")

                        # Display table with both values of Max_depth
                        st.write(df_metrics)

                        # Display metrics with delta comparison
                        st.subheader("Metrics with Delta Comparison")
                        colA, colB = st.columns(2)
                        for index, row in df_metrics.iterrows():
                            metric = row["Metric"]
                            value_none = row["Max_depth = None"]
                            value_4 = row["Max_depth = 4"]

                            # Calculate delta
                            delta = value_4 - value_none

                            # Display two metrics per row
                            if index % 2 == 0:
                                with colA:
                                    st.metric(label=metric, value=f"{value_4:.2f}", delta=f"{delta:.2f}")
                            else:
                                with colB:
                                    st.metric(label=metric, value=f"{value_4:.2f}", delta=f"{delta:.2f}")

    # Subtab: Simulation
    with sub_tabs[1]:
        st.subheader("Simulation")
        st.write("Here we need to add fancy stuff..!!")


# Data for metrics
    


# Confidentiality Note
st.markdown("---")
st.write(
    "\*\*Confidentiality Note:\*\* This application is for internal use only and contains sensitive data. Do not distribute without proper authorization."
)

st.markdown(
    """
    <style>
        .footer {
            width: 100%;
            text-align: center;
            padding: 10px;
            font-size: 14px;
            position: relative;
            bottom: 0;
            margin-top: 50px;
        }

        /* Default footer styles for dark theme */
        .footer.light-theme {
            background-color: white;
            color: grey;
        }

        /* Footer styles for dark theme */
        .footer.dark-theme {
            background-color: #333;
            color: white;
        }

        /* Sticky footer style for page with small content */
        body {
            display: flex;
            flex-direction: column;
            min-height: 100vh;
            justify-content: space-between;
        }
        
    </style>
    <div id="footer" class="footer">
        &copy; Copyright by SSSIHL CADS @2025
    </div>
    <script>
        // Detect if the theme is dark or light and apply the appropriate class
        const theme = window.matchMedia("(prefers-color-scheme: dark)").matches ? 'dark-theme' : 'light-theme';
        document.getElementById("footer").classList.add(theme);
    </script>
    """,
    unsafe_allow_html=True
)


