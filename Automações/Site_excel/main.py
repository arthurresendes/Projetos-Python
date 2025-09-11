import pandas as pd
import streamlit as st

st.set_page_config(page_title="Orçamento")

with st.container():
    st.title("Tabela valores")
    df = pd.read_excel("Treinamento.xlsx" , skiprows=1)
    st.write(df)
