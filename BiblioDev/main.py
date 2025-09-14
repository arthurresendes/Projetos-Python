class Livro:
    def __init__(self, título: str, autor: str, ano: int, disponibilidade: str) -> None:
        self.__titulo = título
        self.__autor = autor
        self.__ano = ano
        self.__disponibilidade = disponibilidade

    def __str__(self) -> str:
        return f"{self.__titulo} | {self.__autor} | {self.__ano} | {self.__disponibilidade}"

    # Getters
    def get_titulo(self) -> str:
        return self.__titulo
    
    def get_autor(self) -> str:
        return self.__autor
    
    def get_ano(self) -> int:
        return self.__ano
    
    def get_disponibilidade(self) -> str:
        return self.__disponibilidade
    
    # Setters
    def set_disponibilidade(self, status: str) -> None:
        self.__disponibilidade = status


listaLivros = []

def menu() -> None:
    while True:
        print("\n---- MENU ----")
        print("1 - Cadastrar Livro")
        print("2 - Listar Livros")
        print("3 - Emprestar")
        print("4 - Devolver")
        print("5 - Remover biblioteca")
        print("0 - Sair")
        print("------------------")
        
        try:
            opcao = int(input(": "))
            
            if opcao == 0:
                print("Até mais...")
                break
            elif opcao == 1:
                titulo = input("Qual título do livro: ").title()
                autor = input("Qual autor do livro: ").title()
                ano = int(input("Qual ano do livro: "))
                disponibilidade = 'Livre'
                livro = Livro(titulo, autor, ano, disponibilidade)
                listaLivros.append(livro)
                print("Livro cadastrado com sucesso!")
            elif opcao == 2:
                if len(listaLivros) == 0:
                    print("Sem livros")
                else:
                    for indice, livro in enumerate(listaLivros):
                        print(f"{indice + 1} - {livro}")
            elif opcao == 3:
                if len(listaLivros) == 0:
                    print("Sem livros")
                else:
                    nome = input("Qual livro quer pegar emprestado: ").title()
                    encontrado = False
                    for livro in listaLivros:
                        if livro.get_titulo() == nome:
                            encontrado = True
                            if livro.get_disponibilidade() == "Livre":
                                livro.set_disponibilidade("Reservado")
                                print("Livro emprestado com sucesso!")
                            else:
                                print("Livro já está emprestado!")
                            break
                    
                    if not encontrado:
                        print("Livro não encontrado!")
            elif opcao == 4:
                if len(listaLivros) == 0:
                    print("Sem livros")
                else:
                    nomeDevolve = input("Qual livro quer devolver: ").title()
                    encontrado = False
                    for livro in listaLivros:
                        if livro.get_titulo() == nomeDevolve:
                            encontrado = True
                            if livro.get_disponibilidade() == "Reservado":
                                livro.set_disponibilidade('Livre')
                                print("Livro devolvido com sucesso!")
                            elif livro.get_disponibilidade() == "Livre":
                                print("Livro já estava livre")
                            break
                    
                    if not encontrado:
                        print("Livro não encontrado!")
            elif opcao == 5:
                if len(listaLivros) == 0:
                    print("Sem livros")
                else:
                    nomeRemove = input("Qual livro quer remover: ").title()
                    encontrado = False
                    for i, livro in enumerate(listaLivros):
                        if livro.get_titulo() == nomeRemove:
                            encontrado = True
                            del listaLivros[i]
                            print("Livro removido com sucesso!")
                            break
                    
                    if not encontrado:
                        print("Livro não encontrado!")
            else:
                print("Opção inválida!")
                
        except ValueError:
            print("Erro: Digite um número válido!")
        except Exception as e:
            print(f"Erro inesperado: {e}")


if __name__ == "__main__":
    menu()