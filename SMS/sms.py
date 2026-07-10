from twilio.rest import Client
from dotenv import load_dotenv
import os

load_dotenv()

user_sid = os.getenv("USER_SID")
token = os.getenv("AUTH_TOKEK_TWILIO")

client = Client(user_sid,token)

message = client.messages.create(
    to=os.getenv("TO"),
    from_=os.getenv("FROM"),
    body="Olá, tudo bem ?"
)
#print(message.sid)