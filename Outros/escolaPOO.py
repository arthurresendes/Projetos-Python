class Pessoa():
    def __init__(self,nome: str,idade: int,cpf: str) -> None:
        self.nome = nome
        self.idade = idade
        self.cpf = cpf
    
    def descricao(self) -> str:
        return f"A pessoa com {self.nome} e com {self.idade} anos tem o seu cpf como: {self.cpf}"


class Aluno(Pessoa):
    def __init__(self,nome:str,idade:int,ano_escolar: int,ensino:str):
        self.ano_escolar = ano_escolar
        self.ensino = ensino
        super().__init__(nome,idade)
    
    def descricao(self) -> str:
        return f"O aluno(a) com {self.nome} e com {self.idade} anos esta no {self.ano_escolar} ano no ensino {self.ensino}"
    

class Professor(Pessoa):
    def __init__(self,nome:str,idade:int,ano_escolar: int,ensino:str,materia:str):
        self.ano_escolar = ano_escolar
        self.ensino = ensino
        self.materia = materia
        super().__init__(nome,idade)
    
    def descricao(self) -> str:
        return f"O professor/professora com {self.nome} e com {self.idade} anos da aula de {self.materia} no {self.ano_escolar} ano no ensino {self.ensino}"

class Aula():
    