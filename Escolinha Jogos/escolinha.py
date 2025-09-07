import forca
import gameNumbers
import gameRandomico
import pedraPapelTesoura


def menu_principal():
    while True:
        print("\n===== CENTRAL DE JOGOS =====")
        print("1 - Jogo Matemática")
        print("2 - Jogo da Forca")
        print("3 - Pedra , Papel , Tesoura")
        print("4 - Adivinha")
        print("0 - Sair")
        
        escolha = input(": ")

        if escolha == "1":
            gameNumbers.menu()  
        elif escolha == "2":
            forca.jogo()
        elif escolha == "3":
            pedraPapelTesoura.modo_jogo()
        elif escolha == "4":
            gameRandomico.main()
        elif escolha == "0":
            print("Saindo...")
            break
        else:
            print("Opção inválida!")

if __name__ == "__main__":
    menu_principal()