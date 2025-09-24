from fastapi import APIRouter

outher_router = APIRouter(prefix="/chamado", tags=['chamado'])


@outher_router.get("/")
async def vizualizar_chamados():
    return {"chamado": "Indisponivel"}