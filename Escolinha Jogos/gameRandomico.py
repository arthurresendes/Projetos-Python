import random as rd

def jogo(opcao:int):
    contador = 0
    if opcao == 1:
        numberPC = rd.randint(1, 10)
        maximo = 10
    elif opcao == 2:
        numberPC = rd.randint(1, 100)
        maximo = 100
    elif opcao == 3:
        numberPC = rd.randint(1, 10000)
        maximo = 10000
    elif opcao == 4:
        numberPC = rd.randint(1, 100000)
        maximo = 100000
    else:
        return "Erro"
    while contador < 3:
        try:
            numberUser = int(input(f"Fale um número entre 1 e {maximo}: "))
            contador += 1
            if numberUser == numberPC:
                print(f"Você acertou com {contador} tentativa(s)!")
                break
            else:
                if numberUser > numberPC:
                    print(f"Seu número é maior que o sorteado. Tentativas restantes: {3 - contador}")
                else:
                    print(f"Seu número é menor que o sorteado. Tentativas restantes: {3 - contador}")
        except ValueError:
            print("Digite um número válido.")
    else:
        print(f"Suas tentativas acabaram! O número era: {numberPC}")

def main():
    opcao = 0
    while opcao != 5:
            print("-------------------------")
            print("Escolha a opção do jogo")
            print("1- Modo fácil")
            print("2- Modo médio")
            print("3- Modo díficil")
            print("4- Modo extremo")
            print("5- Sair")
            print("-------------------------")
            opcao = int(input())
            if opcao == 5:
                print("Saindo...")
            else:
                jogo(opcao)

if __name__ == "__main__":
    main()