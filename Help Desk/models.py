from sqlalchemy import create_engine, Boolean, String, Integer,Column, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base

db = create_engine("sqllite:///desk.bd")

Base = declarative_base()

class Solicitante(Base):
    __tablename__ = "solicitante"
    
    id_solicitante = Column("id_solicitante",Integer, primary_key=True, autoincrement=True)
    nome_completo = Column("nome_completo", String)
    email = Column("email", String, nullable=False)
    rua = Column("rua", String)
    cidade = Column("cidade", String)
    numero = Column("numero", Integer)
    
    def __init__(self,nome_completo,email,rua,cidade,numero):
        self.nome_completo = nome_completo
        self.email = email
        self.rua = rua
        self.cidade = cidade
        self.numero = numero

class Responsavel(Base):
    __tablename__ = "resposavel"
    
    id_responsavel = Column("id_responsavel",Integer, primary_key=True, autoincrement=True)
    nome_completo = Column("nome_completo", String)
    email = Column("email", String, nullable=False)
    especialidade = Column("especialidade", String)
    
    def __init__(self,nome_completo,email,especialidade):
        self.nome_completo = nome_completo
        self.email = email
        self.especialidade = especialidade

class Chamado(Base):
    __tablename__ = "chamado"
    
    id_chamado = Column("id_chamado",Integer, primary_key=True, autoincrement=True)
    status = Column("status", String)
    data_abertura = Column("data_abertura", DateTime)
    data_fechamento = Column("data_abertura", DateTime)
    titulo = Column("titulo", String)
    descricao = Column("descricao", String)
    id_solicitanteFk = Column("id_solicitanteFk", ForeignKey("solicitante.id_solicitante"))
    id_responsavelFk = Column("id_responsavelFk", ForeignKey("responsavel.id_responsavel"))
    
    def __init__(self,status,data_abertura,data_fechamento,titulo,descricao,id_solicitanteFk,id_responsavelFk):
        self.status = status
        self.data_abertura = data_abertura
        self.data_fechamento =data_fechamento
        self.titulo = titulo
        self.descricao = descricao
        self.id_solicitanteFk = id_solicitanteFk
        self.id_responsavelFk = id_responsavelFk



