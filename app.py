import numpy as np
import pandas as pd                
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.impute import SimpleImputer

LANGUAGES = {
    "English": {
        "title": "Data Cleaning and Visualization Tool",
        "upload_file_prompt": "Upload an Excel or CSV file",
        "original_data_label": "Original Data (First 5 Rows):",
        "cleaned_data_label": "Cleaned Data (First 5 Rows):",
        "data_cleaning_section": "Data Cleaning Options",
        "fill_missing_values_option": "Fill Missing Values",
        "fill_method_label": "Filling Method",
        "remove_duplicates_option": "Remove Duplicate Records",
        "remove_outliers_option": "Remove Outliers",
        "clean_text_option": "Clean Text Data",
        "correct_spelling_option": "Correct Spelling Errors",
        "correct_data_format_option": "Correct Data Formats",
        "apply_binning_option": "Apply Binning",
        "select_column_for_binning": "Select Column for Binning",
        "number_of_bins_label": "Number of Bins",
        "visualization_section": "Visualization Options",
        "histogram_option": "Histogram",
        "select_column_for_histogram": "Select column for Histogram",
        "boxplot_option": "Boxplot",
        "select_column_for_boxplot": "Select column for Boxplot",
        "correlation_heatmap_option": "Correlation Heatmap",
        "download_cleaned_data_option": "Download Cleaned Data",
        "download_image_label": "Download Image",
        "unsupported_file_error": "Unsupported file type",
        "binning_error": "Binning can only be applied to numeric columns.",
    },
}


selected_language = st.sidebar.selectbox("Language", ["English"])
language_text = LANGUAGES[selected_language]

st.sidebar.markdown("**Created by [Prajwal Kanade](https://prajwalk-1.github.io/prajwalk-1/)**")

# Load Data
def load_dataset(file):
    if file is not None:
        file_extension = file.name.split('.')[-1]
        if file_extension == 'csv':
            return pd.read_csv(file)
        elif file_extension in ['xlsx', 'xls']:
            return pd.read_excel(file)
        else:
            st.error(language_text["unsupported_file_error"])
            return None
    return None

# Fill missing values
def handle_missing_values(dataframe, method='mean'):
    if method == 'mean':
        return dataframe.fillna(dataframe.mean(numeric_only=True))
    elif method == 'median':
        return dataframe.fillna(dataframe.median(numeric_only=True))
    elif method == 'mode':
        return dataframe.fillna(dataframe.mode().iloc[0])
    return dataframe

# Remove outliers
def eliminate_outliers(dataframe):
    for column in dataframe.select_dtypes(include=[np.number]).columns:
        q1 = dataframe[column].quantile(0.25)
        q3 = dataframe[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        dataframe = dataframe[(dataframe[column] >= lower_bound) & (dataframe[column] <= upper_bound)]
    return dataframe

# Remove duplicate rows
def remove_duplicate_rows(dataframe):
    return dataframe.drop_duplicates()

# Clean text columns
def clean_text_columns(dataframe):
    for column in dataframe.select_dtypes(include=['object']).columns:
        dataframe[column] = dataframe[column].str.strip().str.lower()
    return dataframe

# Correct spelling
def fix_spelling(dataframe):
    return dataframe.applymap(lambda value: value.replace("teh", "the") if isinstance(value, str) else value)

# Correct data formats
def standardize_data_formats(dataframe):
    for column in dataframe.columns:
        try:
            dataframe[column] = pd.to_datetime(dataframe[column], errors='ignore')
            dataframe[column] = pd.to_numeric(dataframe[column], errors='ignore')
        except Exception:
            pass
    return dataframe

# Apply binning
def perform_binning(dataframe, column_name, bins_count):
    try:
        if column_name in dataframe.select_dtypes(include=[np.number]).columns:
            dataframe[f"{column_name}_binned"] = pd.cut(dataframe[column_name], bins=bins_count, labels=False)
        else:
            st.warning(language_text["binning_error"])
    except Exception as error:
        st.error(f"Error in binning: {error}")
    return dataframe

# Visualization helpers
def save_and_display_chart(chart, filename):
    chart.savefig(filename, format='png')
    st.pyplot(chart)
    with open(filename, "rb") as file:
        st.download_button(
            label=language_text["download_image_label"],
            data=file,
            file_name=filename,
            mime="image/png"
        )

def generate_histogram(dataframe):
    st.subheader(language_text["histogram_option"])
    selected_column = st.selectbox(language_text["select_column_for_histogram"], dataframe.select_dtypes(include=[np.number]).columns)
    chart, ax = plt.subplots()
    dataframe[selected_column].hist(bins=20, ax=ax)
    ax.set_title(f"{language_text['histogram_option']} - {selected_column}")
    save_and_display_chart(chart, f"{selected_column}_histogram.png")

def generate_boxplot(dataframe):
    st.subheader(language_text["boxplot_option"])
    selected_column = st.selectbox(language_text["select_column_for_boxplot"], dataframe.select_dtypes(include=[np.number]).columns)
    chart, ax = plt.subplots()
    sns.boxplot(data=dataframe, x=selected_column, ax=ax)
    ax.set_title(f"{language_text['boxplot_option']} - {selected_column}")
    save_and_display_chart(chart, f"{selected_column}_boxplot.png")

def create_correlation_heatmap(dataframe):
    st.subheader(language_text["correlation_heatmap_option"])
    correlation_matrix = dataframe.select_dtypes(include=[np.number]).corr()
    chart, ax = plt.subplots()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt='.2f', ax=ax)
    ax.set_title(language_text["correlation_heatmap_option"])
    save_and_display_chart(chart, "correlation_heatmap.png")

# Streamlit Application
st.title(language_text["title"])

uploaded_file = st.file_uploader(language_text["upload_file_prompt"], type=["csv", "xlsx", "xls"])
if uploaded_file:
    dataset = load_dataset(uploaded_file)
    if dataset is not None:
        st.write(language_text["original_data_label"])
        st.dataframe(dataset.head())

        # Sidebar: Data Cleaning Options
        st.sidebar.title(language_text["data_cleaning_section"])
        cleaned_data = dataset.copy()

        if st.sidebar.checkbox(language_text["fill_missing_values_option"]):
            method = st.sidebar.selectbox(language_text["fill_method_label"], ['mean', 'median', 'mode'])
            cleaned_data = handle_missing_values(cleaned_data, method=method)

        if st.sidebar.checkbox(language_text["remove_duplicates_option"]):
            cleaned_data = remove_duplicate_rows(cleaned_data)

        if st.sidebar.checkbox(language_text["remove_outliers_option"]):
            cleaned_data = eliminate_outliers(cleaned_data)

        if st.sidebar.checkbox(language_text["clean_text_option"]):
            cleaned_data = clean_text_columns(cleaned_data)

        if st.sidebar.checkbox(language_text["correct_spelling_option"]):
            cleaned_data = fix_spelling(cleaned_data)

        if st.sidebar.checkbox(language_text["correct_data_format_option"]):
            cleaned_data = standardize_data_formats(cleaned_data)

        if st.sidebar.checkbox(language_text["apply_binning_option"]):
            column = st.sidebar.selectbox(language_text["select_column_for_binning"], cleaned_data.columns)
            bins = st.sidebar.slider(language_text["number_of_bins_label"], min_value=2, max_value=20, value=5)
            cleaned_data = perform_binning(cleaned_data, column, bins)

        st.write(language_text["cleaned_data_label"])
        st.dataframe(cleaned_data.head())
        
        # Sidebar: Visualization Options
        st.sidebar.title(language_text["visualization_section"])
        if st.sidebar.checkbox(language_text["histogram_option"]):
            generate_histogram(cleaned_data)

        if st.sidebar.checkbox(language_text["boxplot_option"]):
            generate_boxplot(cleaned_data)

        if st.sidebar.checkbox(language_text["correlation_heatmap_option"]):
            create_correlation_heatmap(cleaned_data)

        # Download Cleaned Data
        csv_data = cleaned_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=language_text["download_cleaned_data_option"],
            data=csv_data,
            file_name="cleaned_data.csv",
            mime="text/csv"
        )
