from random import choice

import time

def modo_jogo():
    introducao()
    print("------------------------------------")
    print("Qual modo quer jogar: ")
    print("1- Você x PC ")
    print("2- Um contra Um")
    print("3- Sair")
    modo = 4

    while modo > 3:
        modo = int(input("Opção: "))
        if modo == 1:
            gamePc()
        elif modo == 2:
            gameUsers()
        elif modo == 3:
            print("Até a proxima!!")
            break
        else:
            print("Digite um número válido!!\n")
    print("------------------------------------")

def introducao():
    print("------------------------------------")
    print("Bem-vindos ao maior jogo da história")
    print("Regras: ")
    print("Quem fizer 5 pontos primeiro vence!!")
    print("Boa sorte!!")
    print("------------------------------------")
    time.sleep(5)


def gameUsers():
    contadorUser1 = 0
    contadorUser2 = 0
    while True:
        user1 = input("User 1 -> Pedra | Papel | Tesoura: ").lower()
        user2 = input("User 2 -> Pedra | Papel | Tesoura: ").lower()

        print(f"{user1} x {user2}")

        if user1 == 'pedra' and user2 == 'papel':
            print("User 2 venceu!!")
            contadorUser2 += 1
        elif user1 == 'pedra' and user2 == 'pedra':
            print("Empate!!")
        elif user1 == 'pedra' and user2 == 'tesoura':
            print("User 1!!")
            contadorUser1 += 1

        elif user1 == 'papel' and user2 == 'papel':
            print("Empate!!")
        elif user1 == 'papel' and user2 == 'tesoura':
            print("User 2 venceu!!")
            contadorUser2 += 1
        elif user1 == 'papel' and user2 == 'pedra':
            print("User 1 venceu!!")
            contadorUser1 += 1

        elif user1 == 'tesoura' and user2 == 'tesoura':
            print("Empate!!")
        elif user1 == 'tesoura' and user2 == 'papel':
            print("User 1 venceu!!")
            contadorUser1 += 1
        elif user1 == 'tesoura' and user2 == 'pedra':
            print("User 2 venceu!!")
            contadorUser2 += 1

        else:
            print("Resposta incompatível!!")

        print(f"Pontos user 1 -> {contadorUser1} \nPontos user 2 -> {contadorUser2}")

        if contadorUser1 == 5:
            print("User 1 Winner!!")
            break
        if contadorUser2 == 5:
            print("User 2 Winner!!")
            break

        opcao = input("Digite 0 se quiser sair ou qualquer outra tecla se deseja continuar: ")
        if opcao == "0":
            print("Até a próxima!")
            break

def gamePc():
    contadorUser = 0
    contadorProg = 0
    while True:
        jogo = ['pedra', 'papel', 'tesoura']
        programa = choice(jogo)
        user = input("Pedra | Papel | Tesoura: ").lower()
        print(f"{user} x {programa}")

        if user == 'pedra' and programa == 'papel':
            print("Você perdeu!!")
            contadorProg += 1
        elif user == 'pedra' and programa == 'pedra':
            print("Empate!!")
        elif user == 'pedra' and programa == 'tesoura':
            print("Você venceu!!")
            contadorUser += 1

        elif user == 'papel' and programa == 'papel':
            print("Empate!!")
        elif user == 'papel' and programa == 'tesoura':
            print("Você perdeu!!")
            contadorProg += 1
        elif user == 'papel' and programa == 'pedra':
            print("Você ganhou!!")
            contadorUser += 1

        elif user == 'tesoura' and programa == 'tesoura':
            print("Empate!!")
        elif user == 'tesoura' and programa == 'papel':
            print("Você ganhou!!")
            contadorUser += 1
        elif user == 'tesoura' and programa == 'pedra':
            print("Você perdeu!!")
            contadorProg += 1

        else:
            print("Resposta incompatível!!")
            
        print(f"Seus pontos -> {contadorUser} \nPontos computador -> {contadorProg}")

        if contadorProg == 5:
            print("Computador Winner!!")
            break
        if contadorUser == 5:
            print("You Winner!!")
            break
        
        opcao = input("Digite 0 se quiser sair ou qualquer outra tecla se deseja continuar: ")
        if opcao == "0":
            print("Até a próxima!")
            break

if __name__ == "__main__":
    modo_jogo()