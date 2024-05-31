import streamlit as st
import pandas as pd
import os
from ydata_profiling import ProfileReport
from streamlit_ydata_profiling import st_profile_report
from pygwalker.api.streamlit import StreamlitRenderer

#ML Stuff
from pycaret.classification import setup as setup_classification, compare_models as compare_models_classification, pull, save_model
from pycaret.regression import setup as setup_regression, compare_models as compare_models_regression

st.set_page_config(
        page_title="Use Pygwalker In Streamlit",
        layout="wide"
    )

with st.sidebar:
    #st.image()
    st.title("AutoStreamML")
    choice = st.radio("Navigation",["Upload","Profiling","ML","Download","Visualizer"])
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
    st.title("Train your model")
    target = st.selectbox("Select your target feature",df.columns)
    problem_type = st.selectbox("Select your problem type", ["Classification", "Regression"])
    if st.button("Train Model"):
        if problem_type == "Classification":
            setup_classification(df, target=target)
            best_model= compare_models_classification()
        elif problem_type == "Regression":
            setup_regression(df, target=target)
            best_model= compare_models_regression()

        compare_df = pull()
        st.info("This is the ML Model")
        st.dataframe(compare_df,use_container_width= True)
        best_model
        save_model(best_model,'best_model')


if choice == "Download":
    st.title("Hurray! You can now download your model")
    with open("best_model.pkl","rb") as f:
        st.download_button("Download the Model",f,"best_model.pkl")

if choice == "Visualizer":

    st.title("Visualize Your Data")
    pyg_app = StreamlitRenderer(df)
    
    pyg_app.explorer()
