lista = []
tarefaConcluida = []
def cadastraLista():
    tarefa = input("Digite o nome da tarefa: ")
    lista.append(tarefa)

def mostraTarefa():
    if len(lista) == 0:
        print("Sem valores na lista...")
    else:
        for indice, tarefa in enumerate(lista):
            print(f"{indice + 1} - {tarefa}")

def concluirTarefa():
    if len(lista) == 0:
        print("Sem tarefas para concluir...")
    else:
        for indice, tarefa in enumerate(lista):
            print(f"{indice + 1} - {tarefa}")
        marcarConcluida = int(input("Digite o número da tarefa que você concluiu: ")) - 1
        # Se marcarConcluida for maior que 0 e menor que o tamanho da lista de tarefas remove a concluida da lista de tarefa e adiciona na lista tarefasConcluida
        if 0 <= marcarConcluida < len(lista):
            tarefa = lista.pop(marcarConcluida)
            tarefaConcluida.append(tarefa)
            print(f"Tarefa concluída: '{tarefa}'")
        else:
            print("Número inválido.")

def mostrarConcluidas():
    if tarefaConcluida:
        print("Tarefas concluídas:")
        for tarefa in tarefaConcluida:
            print(f"- {tarefa}")
    else:
        print("Nenhuma tarefa concluída ainda.")

def removeTarefa():
    valorRemover = int(input("Digite o número da tarefa que quer remover: ")) - 1
    if 0 <= valorRemover < len(lista):
        tarefa_removida = lista.pop(valorRemover)
        print(f"Tarefa '{tarefa_removida}' removida com sucesso!")
    else:
        print("Tarefa inexistente!")

def main():
    opcao = 0
    while opcao != 6:
        print("------------------------")
        print("1- Cadastrar tarefa")
        print("2- Listar tarefa")
        print("3- Remover tarefa")
        print("4- Marcar tarefa concluida")
        print("5- Listar concluidos")
        print("6- Sair")
        print("------------------------")
        opcao = int(input())
        if opcao == 1:
            cadastraLista()
        elif opcao == 2:
            mostraTarefa()
        elif opcao == 3:
            removeTarefa()
        elif opcao == 4:
            concluirTarefa()
        elif opcao == 5:
            mostrarConcluidas()
        elif opcao == 6:
            print("Saindo...")
        else:
            print("Número não estabelecido")

main()