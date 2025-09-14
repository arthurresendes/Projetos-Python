class Conta:
    def __init__(self,titular:str,saldo:float) -> None:
        self.__titular = titular
        self.__saldo = saldo
    
    def sacar(self,quantidade:float) -> None:
        if quantidade > self.__saldo:
            print("Erro: saldo insuficiente")
        else:
            self.__saldo -= quantidade
    
    def depositar(self,valor:float) -> None:
        self.__saldo += valor
    
    def transferir(self, destino, valor: float) -> None:
        if valor > self.__saldo:
            print("Erro: saldo insuficiente para transferência")
        else:
            self.__saldo -= valor
            destino.__saldo += valor
        
    @property
    def ver_saldo(self):
        return f"O saldo do {self.__titular} é de {self.__saldo} R$"
    
    def __str__(self) -> str:
        return f"Titular: {self.__titular} | Saldo: {self.__saldo:.2f} R$"



listaPessoas = []
# -------------------------
# Menu de operações
# -------------------------
def menu():
    listaPessoas = []

    while True:
        print("\n---- MENU ----")
        print("1 - Listar contas")
        print("2 - Sacar")
        print("3 - Depositar")
        print("4 - Transferir")
        print("5 - Adicionar pessoa")
        print("6 - Ver saldo")
        print("0 - Sair")
        print("------------------")
        
        try:
            escolha = int(input("Escolha uma opção: "))
            
            if escolha == 0:
                print("Saindo... até logo!")
                break

            elif escolha == 1:  # LISTAR CONTAS
                if len(listaPessoas) == 0:
                    print("Lista vazia!!")
                else:
                    for indice, pessoa in enumerate(listaPessoas, start=1):
                        print(f"{indice} - {pessoa}")

            elif escolha == 2:  # SACAR
                if len(listaPessoas) == 0:
                    print("Lista vazia!!")
                else:
                    print("---- As buscas são feitas pelo indice da pessoa não pelo nome ----")
                    saque = int(input("Digite o índice da conta: "))
                    valor = float(input("Qual valor deseja sacar: "))
                    listaPessoas[saque-1].sacar(valor)

            elif escolha == 3:  # DEPOSITAR
                if len(listaPessoas) == 0:
                    print("Lista vazia!!")
                else:
                    print("---- As buscas são feitas pelo indice da pessoa não pelo nome ----")
                    dep = int(input("Digite o índice da conta: "))
                    valor = float(input("Qual valor deseja depositar: "))
                    listaPessoas[dep-1].depositar(valor)

            elif escolha == 4:  # TRANSFERIR
                if len(listaPessoas) < 2:
                    print("É preciso ter pelo menos 2 contas para transferir!")
                else:
                    print("---- As buscas são feitas pelo indice da pessoa não pelo nome ----")
                    origem = int(input("De qual conta vai sair o valor: "))
                    destino = int(input("Para qual conta vai a transferência: "))
                    valor = float(input("Qual valor deseja transferir: "))
                    listaPessoas[origem-1].transferir(listaPessoas[destino-1], valor)
            
            elif escolha == 5:  # ADICIONAR PESSOA
                nome = input("Digite seu nome: ").title()
                sald = float(input("Digite o saldo: "))
                pessoa = Conta(nome, sald)
                listaPessoas.append(pessoa)
                print(f"Conta criada com sucesso para {nome}!")
            
            elif escolha == 6: # VER SALDO
                if len(listaPessoas) == 0:
                    print("É preciso ter pelo menos 2 contas para transferir!")
                else:
                    saldoPessoa = int(input("Qual conta quer ver o saldo: "))
                    print(listaPessoas[saldoPessoa-1].ver_saldo)
            else:
                print("Opção inválida!")

        except Exception as e:
            print("Erro, tente novamente!", e)


if __name__ == "__main__":
    menu()