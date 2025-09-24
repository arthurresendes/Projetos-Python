from fastapi import FastAPI, APIRouter

app = FastAPI()
# uvicorn main:app --reload

from auth_routes import auth_router
from outher_routes import outher_router

app.include_router(auth_router)
app.include_router(outher_router)