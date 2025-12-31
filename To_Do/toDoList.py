class Todo:
    def __init__(self,lista:list):
        self.__lista = lista
        
    def cadastro(self,tarefa:str):
        self.__lista.append(tarefa)
        return "Tarefa cadastrada"
    
    def ver_lista(self):
        cont = 1
        for i in self.__lista:
            print(f"{[cont]}: {i}")
            cont +=1
    
    def atualizar(self,tarefaAtualizada: str, indice: int):
        self.__lista[indice - 1] = tarefaAtualizada
        return self.ver_lista()
    
    def deletar(self, indice: int):
        del self.__lista[indice-1]
        return self.ver_lista()

    def __len__(self):
        return len(self.__lista)


if __name__ == "__main__":
    lista_tarefas = []
    meuTodo = Todo(lista_tarefas)
    while True:
        try:
            print("\n1- Adicionar Tarefa\n2- Atualizar Tarefa\n3- Lista Tarefas\n4- Deletar Tarefa\n5- Sair\n")
            opcao = int(input("Escolha:"))
            if opcao == 1:
                tarefa = input("Digite sua tarefa: ")
                meuTodo.cadastro(tarefa)
            elif opcao == 2:
                while True:
                    indice = int(input("Qual indice quer atualizar: "))
                    if indice > meuTodo.__len__() or indice < 1:
                        print("Digite um indice valido!!")
                    else:
                        break
                tarefa = input("Digite o nome novo da tarefa: ")
                meuTodo.atualizar(tarefa,indice)
            elif opcao == 3:
                meuTodo.ver_lista()
            elif opcao == 4:
                while True:
                    indice = int(input("Qual indice quer deletar: "))
                    if indice > meuTodo.__len__() or indice < 1:
                        print("Digite um indice valido!!")
                    else:
                        meuTodo.deletar(indice)
                        break
            elif opcao ==5:
                print("Saindo.")
                break
        except:
            print("Escolha um valor valido !!")