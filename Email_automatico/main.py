import win32com.client as win32
import os

# Configuração do Outlook
outlook = win32.Dispatch('outlook.application')

lista_certificacao = [
    {
        "nome": "Arthur Resende",
        "email": "arthur.resende.gomes02@gmail.com"
    }
]

caminho_foto = os.path.abspath(
    r"caminhoImagem"
)

for pessoa in lista_certificacao:
    email = outlook.CreateItem(0)
    email.To = pessoa["email"]
    email.CC = "josezin"
    email.Subject = f"Parabens, {pessoa['nome']} você terminou o curso!"

    anexo = email.Attachments.Add(caminho_foto)
    anexo.PropertyAccessor.SetProperty(
        "http://schemas.microsoft.com/mapi/proptag/0x3712001F",
        "certificado"
    )

    email.HTMLBody = f"""
    <html>
        <body style="font-family: Arial, sans-serif; color: #333;text-align:center;">
            <h2 style="color: #d4418e;">Parabéns, {pessoa['nome']}!</h2>
            <p>Muito bem em concluir o curso</p>
            <p>Que este novo ciclo seja incrível!</p>
            <br>
            <img src="cid:certificado" width="400">
            <br>
            <p>Com carinho,<br><strong>Empresa</strong></p>
        </body>
    </html>
    """

    try:
        email.Send()
        print(f"Parabéns enviado para {pessoa['nome']} ({pessoa['email']})")
    except Exception as e:
        print(f"Erro ao enviar para {pessoa['nome']}: {e}")

print("\nProcesso de envio finalizado!")