import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

cliente = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def chatgpt(prompt_text):
    try:
        resposta = cliente.chat.completions.create(
            model="gpt-5o-mini",
            messages=[
                {"role": "system", "content": "Você é um assistente prestativo e busca explicar da melhor maneira possível, onde qualquer tipo de pessoa entenda."},
                {"role": "user", "content": prompt_text}
            ],
            max_tokens=150,
            temperature=0.7,
        )
        
        return resposta.choices[0].message.content.strip()
    
    except Exception as e:
        return f"Ocorreu um erro: {e}"


if __name__ == "__main__":
    while True:
        prompt = input("Digite sair para sair!! \nFaça sua pergunta: ").lower()
        if prompt == 'sair':
            print("Saindo...")
            break
        else:
            resposta = chatgpt(prompt)
            if "Ocorreu um erro" in resposta:
                print("Sem creditos em sua API, recarregue devidamenete")
                print("Saindo...")
                break
            else:
                print(f"Resposta do assistente: {resposta}")

