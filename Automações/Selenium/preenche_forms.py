from selenium import webdriver
import streamlit as st
from selenium.webdriver.common.by import By
import time

def main():
    st.title("Olá, bem-vindo a uma automação com o Selenium + Streamlit")
    st.subheader("Esse é apenas um modo de mostrar o poder das bibliotecas")
    st.subheader("Como irá funcionar: ")
    st.write("Tem que preencher corretamente os campos de nome , email e mensagem (telefone não é obrigatorio) e irá preencher automaticamente o formulario do meu portfolio com as suas informações + mensagem e enviar para o meu e-mail !!")
    
    nome = st.text_input("Digite seu nome: ")
    email = st.text_input("Digite seu emaiL: ")
    telefone = st.text_input("Digite seu telefone: ")
    mensagem = st.text_area("Digite sua mensagem: ")
    botao = st.button("Enviar")
    if botao:
        if not nome:
            st.error("O campo 'Nome' é obrigatório.")
        elif not email or not "@gmail.com" in  email:
            st.error("O campo 'E-mail' é obrigatório com @gmail.com.")
        elif not mensagem:
            st.error("O campo 'Mensagem' é obrigatório.")
        else:
            try:
                driver = webdriver.Chrome()
                driver.get("https://arthurresendes.github.io/Portfolio/portfolio2.html")
                time.sleep(1)
                campo_nome = driver.find_element(By.ID, "nome")
                campo_nome.send_keys(nome)
                
                campo_email = driver.find_element(By.ID, "email")
                campo_email.send_keys(email)
                
                campo_telefone = driver.find_element(By.ID, "telefone")
                campo_telefone.send_keys(telefone)
                
                campo_mensagem = driver.find_element(By.ID, "msg")
                campo_mensagem.send_keys(mensagem)
                
                enviar = driver.find_element(By.CSS_SELECTOR, "input[value='ENVIAR']")
                enviar.click()
                
                time.sleep(2)
                driver.close()
                st.success("O seu envio foi um sucesso!!")
            except:
                st.error("Erro ao executar!!")    


if __name__ == "__main__":
    main()