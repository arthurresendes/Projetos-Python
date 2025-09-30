from fastapi import APIRouter,Depends
from models import Solicitante, Responsavel
from dependencias import pegar_sessao


auth_router = APIRouter(prefix="/auth", tags=['auth'])

@auth_router.get("/")
async def padra():
    return {"mensagem": "Olá , você esta no help desk"}

@auth_router.post("/verificacao_conta_solicitante")
async def verifica_solicitante(nome:str,email:str, session = Depends(pegar_sessao)):
    verificacaoSolicitante =  session.query(Solicitante).filter(Solicitante.email == email).first()
    if verificacaoSolicitante:
        return {"mensagem": "Email encontrado e validado"}
    else:
        novo_solicitante = Solicitante(nome,email)
        session.add(novo_solicitante)
        session.commit()
        return {"mensagem": "Usuario cadastrado}"}

@auth_router.post("/verificacao_conta_responsavel")
async def verifica_solicitante(nome:str,email:str,especialidade = str, session = Depends(pegar_sessao)):
    verificacaoSolicitante =  session.query(Responsavel).filter(Responsavel.email == email).first()
    if verificacaoSolicitante:
        return {"mensagem": "Email de responsavel encontrado"}
    else:
        novo_responsavel = Responsavel(nome,email,especialidade)
        session.add(novo_responsavel)
        session.commit()
        return {"mensagem": "Responsavel cadastrado}"}