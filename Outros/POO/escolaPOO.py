class Pessoa():
    def __init__(self,nome: str,idade: int) -> None:
        self.nome = nome
        self.idade = idade
    
    def descricao(self) -> str:
        return f"A pessoa com {self.nome} e com {self.idade} anos"


class Aluno(Pessoa):
    def __init__(self,nome:str,idade:int,ano_escolar: int,ensino:str):
        self.ano_escolar = ano_escolar
        self.ensino = ensino
        super().__init__(nome,idade)
    
    def descricao(self) -> str:
        return f"O aluno(a) com nome de {self.nome} e com {self.idade} anos esta no {self.ano_escolar} ano no ensino {self.ensino}"
    

class Professor(Pessoa):
    def __init__(self,nome:str,idade:int,ano_escolar: int,ensino:str,materia:str):
        self.ano_escolar = ano_escolar
        self.ensino = ensino
        self.materia = materia
        super().__init__(nome,idade)
    
    def descricao(self) -> str:
        return f"O professor/professora com o nome de {self.nome} tem {self.idade} anos, da aula de {self.materia} no {self.ano_escolar} ano no ensino {self.ensino}"

class Aula():
    def __init__(self,aluno:Aluno, professor: Professor):
        self.aluno = aluno
        self.professor = professor
    
    def ver_relacao_aluno_professor(self):
        if self.aluno.ano_escolar == self.professor.ano_escolar and self.aluno.ensino.lower() == self.professor.ensino.lower():
            return f"O professor {self.professor.nome} da aula para o aluno(a) {self.aluno.nome} na materia {self.professor.materia}"
        else:
            return  f"O professor {self.professor.nome} não da aula para o aluno(a) {self.aluno.nome}"


if __name__ == "__main__":
    aluno1 = Aluno('Arthur Resende', 16, 2, 'Medio')
    print(aluno1.descricao())
    professor1 = Professor('Sandra', 53, 3, 'Medio', 'Matematica')
    print(professor1.descricao())
    professor2 = Professor('Marcus Todday' , 61, 2, 'Medio', 'Fisica')
    print(professor2.descricao())
    
    aula1 = Aula(aluno1,professor1)
    print(aula1.ver_relacao_aluno_professor())
    aula2 = Aula(aluno1,professor2)
    print(aula2.ver_relacao_aluno_professor())