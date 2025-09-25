from openai import OpenAI
from dotenv import load_dotenv
import os

# Carregar variáveis do .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Criar cliente
client = OpenAI(api_key=api_key) # Sua API carrega aqui

def criar_chatbot(mensagem):
    resposta = client.chat.completions.create(
        model="gpt-5",
        messages=[{"role": "user", "content": mensagem}],
        max_tokens=150,
        temperature=0.7
    )
    return resposta.choices[0].message.content.strip()

if __name__ == "__main__":
    print("Bem-vindo ao Chatbot!!")
    while True:
        mensagem_usuario = input("Digite sua mensagem: ")
        if mensagem_usuario.lower() in ["sair", "exit", "quit"]:
            print("Chatbot: Até mais!")
            break
        resposta_chatbot = criar_chatbot(mensagem_usuario)
        print(resposta_chatbot)
