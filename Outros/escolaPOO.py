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
    def __init__(self,aluno:Aluno, professor: Professor):
        self.aluno = aluno
        self.professor = professor
    
    def ver_aluno_professor(self):
        if self.aluno.ano_escolar == self.professor.ano_escolar:
            return f"O professor {self.professor.nome} da aula para o aluno(a) {self.aluno} na materia {self.professor.materia}"
        else:
            return  f"O professor não {self.professor.nome} da aula para o aluno(a) {self.aluno}"


if __name__ == "__main__":
    