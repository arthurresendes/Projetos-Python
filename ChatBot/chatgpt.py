import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_text(prompt_text):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Você é um assistente prestativo e conciso."},
                {"role": "user", "content": prompt_text}
            ],
            max_tokens=150,
            temperature=0.7,
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        return f"Ocorreu um erro: {e}"


if __name__ == "__main__":
    while True:
        prompt = input("Digite sair para sair!! \nFaça sua pergunta: ").lower()
        if prompt == 'sair':
            break
        else:
            resposta = generate_text(prompt)
            if "Ocorreu um erro" in resposta:
                print("Sem creditos em sua API, recarregue devidamenete")
                break
            else:
                print(f"Resposta do assistente: {resposta}")

