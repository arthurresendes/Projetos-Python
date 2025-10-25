class Pessoa():
    def __init__(self,nome: str,idade: int,cpf: str) -> None:
        self.nome = nome
        self.idade = idade
        self.cpf = cpf
    
    def descricao(self) -> str:
        return f""