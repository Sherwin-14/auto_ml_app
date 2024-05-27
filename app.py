import streamlit as st
import pandas as pd


with st.sidebar:
    #st.image()
    st.title("AutoStreamML")
    choice = st.radio("Navigation",["Upload","Profiling","ML","Download"])
    st.info("This application allows you to build an automated ML pipeline using Streamlit, Pandas Profiling, PyCaret")

if choice == "Upload":
    st.title("Upload Your Data for Modelling!")

    file = st.file_uploader("Upload Your Dataset")

    if file:
        df = pd.read_csv(file)
        st.dataframe(df)
        pass

if choice == "Profiling":
    pass


if choice == "ML":
    pass


if choice == "Download":
    pass
