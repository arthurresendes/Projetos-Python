from random import choice
import time

def menu():
    print("------------------------------------")
    print("Bem-vindo ao jogo da forca")
    print("Você tera 5 chances para acertar!!")
    print("-----------------------------------")
    time.sleep(2)

def jogo():
    menu()
    lista = ['jabuticaba', 'abacaxi', 'banana','limao','maça', 'laranja']
    escolha = choice(lista)  
    letras_user = []         
    chances = 5            
    ganhou = False
    while True:
        for letra in escolha:
            if letra in letras_user:
                print(letra, end=" ")
            else:
                print("_", end=" ")
        print('\n')
        tentativa = input("Digite uma letra: ").lower()

        if tentativa in letras_user:
            print("Você já tentou essa letra!")
            continue

        letras_user.append(tentativa)

        if tentativa not in escolha.lower():
            chances -= 1
            print(f"Total de chances: {chances}")

        ganhou = True
        for letra in escolha:
            if letra.lower() not in letras_user: 
                ganhou = False

        if chances == 0 or ganhou:
            break

    if ganhou:
        print(f"Parabéns, você ganhou! A palavra era {escolha}.")
    else:
        print(f"Você perdeu! A palavra era {escolha}.")

if __name__ == "__main__":
    jogo()