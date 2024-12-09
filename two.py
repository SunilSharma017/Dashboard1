import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
@st.cache
def load_data():
    return sns.load_dataset("iris")

data = load_data()

# Sidebar filters
st.sidebar.title("Filters")
species_filter = st.sidebar.multiselect(
    "Select Species:",
    options=data['species'].unique(),
    default=data['species'].unique()
)

# Filter data
filtered_data = data[data['species'].isin(species_filter)]

# Display filtered data
st.subheader("Filtered Data")
st.write(filtered_data)

# Visualization
st.subheader("Sepal Length vs. Sepal Width")
fig, ax = plt.subplots()
sns.scatterplot(data=filtered_data, x="sepal_length", y="sepal_width", hue="species", ax=ax)
st.pyplot(fig)

# Histogram
st.subheader("Distribution of Petal Length")
fig, ax = plt.subplots()
sns.histplot(data=filtered_data, x="petal_length", kde=True, hue="species", ax=ax)
st.pyplot(fig)
