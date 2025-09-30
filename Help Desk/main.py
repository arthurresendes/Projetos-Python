from fastapi import FastAPI
from passlib.context import CryptContext
from dotenv import load_dotenv
import os

load_dotenv()
SECRET_KEY = os.getenv("SECRET_KEY")


app = FastAPI()
# uvicorn main:app --reload

bcrypt_context = CryptContext(schemes=["bcrypt"], deprecated=["auto"])

from auth_routes import auth_router
from outher_routes import outher_router

app.include_router(auth_router)
app.include_router(outher_router)