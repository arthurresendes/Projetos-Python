import pywhatkit as kit
from datetime import datetime

data_atual = datetime.now()

'''
Vale lembrar que isso é so um molde para mostrar conhecimento
'''


# Colocar o número da pessoa e a mensagem, mensagem instantanea
kit.sendwhatmsg_instantly("+XXXXXXXXXXXXX",
                            "Bom dia , quais novidades do dia")

# Mensagem programada
kit.sendwhatmsg("+XXXXXXXXXXXXX",
                            "Biom dia quais novidades do dia",
                            data_atual.hour,
                            data_atual.minute + 2)

# Mensagem para email
kit.send_mail("email_sender", "senha_app", "assunto", "mensagem", "destinatarios")