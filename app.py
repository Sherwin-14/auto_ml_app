import streamlit as st
import pandas as pd
import os
from ydata_profiling import ProfileReport
from streamlit_ydata_profiling import st_profile_report

with st.sidebar:
    #st.image()
    st.title("AutoStreamML")
    choice = st.radio("Navigation",["Upload","Profiling","ML","Download"])
    st.info("This application allows you to build an automated ML pipeline using Streamlit, Pandas Profiling, PyCaret")

if os.path.exists("source_data.csv"):
    df = pd.read_csv("source_data.csv",index_col=None)    

if choice == "Upload":
    st.title("Upload Your Data for Modelling!")

    file = st.file_uploader("Upload Your Dataset")

    if file:
        df = pd.read_csv(file,index_col=None)
        df.to_csv("source_data.csv",index=None)
        st.dataframe(df)
        pass

if choice == "Profiling":
    st.title("Automated EDA")
    pr = ProfileReport(df,minimal=True, orange_mode=True, explorative=True)

    st_profile_report(pr, navbar=True)


if choice == "ML":
    pass


if choice == "Download":
    pass
