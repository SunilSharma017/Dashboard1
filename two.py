import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Cache data loading to optimize performance
@st.cache_data
def load_data():
    iris_data = load_iris()
    iris_df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
    iris_df['species'] = pd.Categorical.from_codes(iris_data.target, iris_data.target_names)
    return iris_df

# Load Iris dataset
iris_df = load_data()

# Sidebar for user interaction
st.sidebar.header("Iris Dataset Dashboard")
st.sidebar.write("Explore the Iris dataset interactively.")
visualization = st.sidebar.selectbox("Select Visualization", ("Pairplot", "Scatter Plot", "Box Plot", "Bar Chart"))

# Title and Description
st.title("Iris Dataset Dashboard")
st.write("This dashboard allows you to explore the famous Iris dataset interactively.")

# Display the dataset
st.subheader("Dataset Overview")
if st.checkbox("Show dataset"):
    st.dataframe(iris_df)

# Visualizations
if visualization == "Pairplot":
    st.subheader("Pairplot")
    st.write("Explore the relationships between features.")
    species_filter = st.multiselect("Select species to include", options=iris_df['species'].unique(), default=iris_df['species'].unique())
    filtered_df = iris_df[iris_df['species'].isin(species_filter)]
    fig = sns.pairplot(filtered_df, hue="species", palette="Set2")
    st.pyplot(fig)

elif visualization == "Scatter Plot":
    st.subheader("Scatter Plot")
    st.write("Visualize feature relationships with scatter plots.")
    x_axis = st.selectbox("X-axis feature", iris_df.columns[:-1])
    y_axis = st.selectbox("Y-axis feature", iris_df.columns[:-1], index=1)
    species_filter = st.multiselect("Select species to include", options=iris_df['species'].unique(), default=iris_df['species'].unique())
    filtered_df = iris_df[iris_df['species'].isin(species_filter)]
    fig, ax = plt.subplots()
    sns.scatterplot(data=filtered_df, x=x_axis, y=y_axis, hue="species", palette="Set2", ax=ax)
    st.pyplot(fig)

elif visualization == "Box Plot":
    st.subheader("Box Plot")
    st.write("Explore the distribution of features.")
    feature = st.selectbox("Select feature", iris_df.columns[:-1])
    fig, ax = plt.subplots()
    sns.boxplot(data=iris_df, x="species", y=feature, palette="Set2", ax=ax)
    st.pyplot(fig)

elif visualization == "Bar Chart":
    st.subheader("Bar Chart")
    st.write("Compare mean values of features across species.")
    feature = st.selectbox("Select feature", iris_df.columns[:-1])
    mean_values = iris_df.groupby("species")[feature].mean().reset_index()
    fig, ax = plt.subplots()
    sns.barplot(data=mean_values, x="species", y=feature, palette="Set2", ax=ax)
    st.pyplot(fig)

# Footer
st.write("Dashboard created with ❤️ using Streamlit.")
