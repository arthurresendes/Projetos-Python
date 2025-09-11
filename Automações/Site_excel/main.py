import pandas as pd
import streamlit as st

df = pd.read_excel("Treinamento.xlsx")
print(df.head())
filtro = df[df['Janeiro']].head()
print(filtro)