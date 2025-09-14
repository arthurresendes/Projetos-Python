perguntas = [
    "Você esteve no local do crime? (S/N): ",
    "Você conhecia a vítima? (S/N): ",
    "Você já teve alguma desavença com a vítima? (S/N): ",
    "Você possui arma de fogo? (S/N): ",
    "Você mentiu em alguma das respostas? (S/N): "
] 

respostas = []
contador = 0

for pergunta in perguntas:
    resposta = input(pergunta).upper()
    respostas.append(resposta)
    if resposta == 'S':
        contador += 1

print("\nRespostas registradas:", respostas)
print(f"Total de respostas 'S': {contador}")

if contador == 1:
    print("Obrigado pelas respostas!!")
elif contador == 2:
    print("Estamos de olho!!")
elif contador == 3:
    print("Suspeita!!")
else:
    print("Suspeita Máxima!!")