from random import choice
import time

contadorUser1 = 0
contadorUser2 = 0
contadorUser = 0
contadorProg = 0

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

def result(result: str, modo: str):
    global contadorUser1
    global contadorUser2
    global contadorUser
    global contadorProg
    
    if modo == "persons":
        if result == "pedra x tesoura" or  result == "papel x pedra" or result == "tesoura x papel":
            contadorUser1+=1
            print("User 1 venceu")
        elif result == "tesoura x pedra" or  result == "pedra x papel" or result == "papel x tesoura":
            contadorUser2 += 1
            print("User 2 venceu")
        else:
            print("Resposta incopativel")
    else:
        if result == "pedra x tesoura" or  result == "papel x pedra" or result == "tesoura x papel":
            contadorUser+=1
            print("Você venceu")
        elif result == "tesoura x pedra" or  result == "pedra x papel" or result == "papel x tesoura":
            contadorProg += 1
            print("Você perdeu")
        else:
            print("Resposta incopativel")

def gameUsers():
    global contadorUser1
    global contadorUser2
    while True:
        user1 = input("User 1 -> Pedra | Papel | Tesoura: ").lower()
        user2 = input("User 2 -> Pedra | Papel | Tesoura: ").lower()

        if user1 == user2:
            print("Empate")
        else:
            res = f"{user1} x {user2}"
            result(res,"persons")

        print(f"Pontos user 1 -> {contadorUser1} \nPontos user 2 -> {contadorUser2}")

        if contadorUser1 == 5:
            print("User 1 Winner!!")
            break
        if contadorUser2 == 5:
            print("User 2 Winner!!")
            break

        opcao = input("Digite 0 se quiser sair ou qualquer outra tecla se deseja continuar: ")
        if opcao == "0":
            print("Jogo encerrado... Até a próxima!")
            break

def gamePc():
    global contadorUser
    global contadorProg

    while True:
        jogo = ['pedra', 'papel', 'tesoura']
        programa = choice(jogo)
        user = input("Pedra | Papel | Tesoura: ").lower()
        if user == programa:
            print("Empate")
        else:
            res = f"{user} x {programa}"
            result(res,"compuPersons")

        print(f"Seus pontos -> {contadorUser} \nPontos computador -> {contadorProg}")

        if contadorProg == 5:
            print("Computador Winner!!")
            break
        if contadorUser == 5:
            print("You Winner!!")
            break
        
        opcao = input("Digite 0 se quiser sair ou qualquer outra tecla se deseja continuar: ")
        if opcao == "0":
            print("Jogo encerrado... Até a próxima!")
            break

if __name__ == "__main__":
    modo_jogo()