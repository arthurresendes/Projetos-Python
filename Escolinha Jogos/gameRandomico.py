import random as rd

def jogoFacil():
    contador = 0
    numberPC = rd.randint(1, 10)
    while contador < 5:
        try:
            numberUser = int(input("Fale um número entre 1 e 10: "))
            contador += 1
            if numberUser == numberPC:
                print(f"Você acertou com {contador} tentativa(s)!")
                break
            else:
                if numberUser > numberPC:
                    print(f"Seu número é maior que o sorteado. Tentativas restantes: {5 - contador}")
                else:
                    print(f"Seu número é menor que o sorteado. Tentativas restantes: {5 - contador}")
        except ValueError:
            print("Digite um número válido.")
    else:
        print(f"Suas tentativas acabaram! O número era: {numberPC}")

def jogoMedio():
    contador = 0
    numberPC = rd.randint(1, 100)
    while contador < 5:
        try:
            numberUser = int(input("Fale um número entre 1 e 100: "))
            contador += 1
            if numberUser == numberPC:
                print(f"Você acertou com {contador} tentativa(s)!")
                break
            else:
                if numberUser > numberPC:
                    print(f"Seu número é maior que o sorteado. Tentativas restantes: {5 - contador}")
                else:
                    print(f"Seu número é menor que o sorteado. Tentativas restantes: {5 - contador}")
        except ValueError:
            print("Digite um número válido.")
    else:
        print(f"Suas tentativas acabaram! O número era: {numberPC}")

def jogoDificil():
    contador = 0
    numberPC = rd.randint(1, 10000)
    while contador < 3:
        try:
            numberUser = int(input("Fale um número entre 1 e 10.000: "))
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

def jogoExtreme():
    contador = 0
    numberPC = rd.randint(1, 100000)
    while contador < 3:
        try:
            numberUser = int(input("Fale um número entre 1 e 100.000: "))
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
            if opcao == 1:
                jogoFacil()
            elif opcao == 2:
                jogoMedio()
            elif opcao == 3:
                jogoDificil()
            elif opcao == 4:
                jogoExtreme()
            elif opcao == 5:
                print("Saindo...")

if __name__ == "__main__":
    main()