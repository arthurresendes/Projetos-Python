class Pessoa():
    def __init__(self,nome: str,idade: int,cpf: str) -> None:
        self.nome = nome
        self.idade = idade
        self.cpf = cpf
    
    def descricao(self) -> str:
        return f"A pessoa com {self.nome} e com {self.idade} anos tem o seu cpf como: {self.cpf}"


class Aluno(Pessoa):
    pass
    

class Professor(Pessoa):
    pass