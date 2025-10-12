import pandas as pd
import streamlit as st
import requests

requisicao = requests.get("https://economia.awesomeapi.com.br/json/last/USD-BRL,EUR-BRL,BTC-BRL")
dic_requisi = requisicao.json()

st.set_page_config(page_title="Conversão")

with st.container():
    st.title("Dolar , Euro e Bitcoin em tempo real")
    st.write("Sua tabela deve conter as colunas sendo chamadas de CONTAS que seriam os nomes das contas e TOTAL que seria o total somatorio dos meses!!")
    st.write("Exemplo de planilha: ")
    df_exemplo = pd.read_excel("Treinamento.xlsx", skiprows=1, skipfooter=1)
    st.dataframe(df_exemplo)
    uploaded_file = st.file_uploader("Escolha um arquivo .xlsx", type=["xlsx"])
    if uploaded_file:
        df = pd.read_excel(uploaded_file , skiprows=1, skipfooter=1)
        dadosUSD = dic_requisi['USDBRL']
        dadosEUR = dic_requisi['EURBRL']
        dadosBTC = dic_requisi['BTCBRL']
        conversaoDOL = dadosUSD['bid']
        conversaoEUR = dadosEUR['bid']
        conversaoBTC = dadosBTC['bid']
        

        st.subheader("Planilha valores: ")
        st.data_editor(df)
        df['Total Dolar'] = df['TOTAL'] / float(conversaoDOL)
        df['Total Euro'] = df['TOTAL'] / float(conversaoEUR)
        df['Total Bitcoin'] = df['TOTAL'] / float(conversaoBTC)
        
        st.subheader(f"Dolar em tempo real: {conversaoDOL}")
        st.write("Conversão: ")
        st.write(df[['CONTAS','Total Dolar']])
        st.write("\n")
        st.subheader(f"Euro em tempo real: {conversaoEUR}")
        st.write("Conversão: ")
        st.write(df[['CONTAS','Total Euro']])
        st.write("\n")
        st.subheader(f"Bitcoin em tempo real: {conversaoBTC}")
        st.write("Conversão: ")
        st.write(df[['CONTAS','Total Bitcoin']])
