import qrcode
import barcode
from barcode.writer import ImageWriter
import validators

def direcinamento():
    while True:
        print("="*50)
        print("1- Gerar código de barra")
        print("2- Gerar QrCode")
        try:
            res = int(input(": "))
            if res == 1 or res == 2:
                break
        except:
            print("Digite um valor valido")
    return res

def gera_cod_barra():
    while True:
        try:
            numero_codigo_str = input("Digite numeros do codigo (12 digitos): ")
            
            if len(numero_codigo_str) == 12 and numero_codigo_str.isdigit():
                break
            
            else:
                print("Erro de digitação! Certifique-se de digitar exatamente 12 dígitos numéricos.")
        except ValueError:
            print("Erro de entrada! Por favor, insira apenas números.")

    ean = barcode.get('ean13', numero_codigo_str, writer=ImageWriter())
    nome_arquivo = ean.save('codigo_de_barras')
    print(f"Código de barras salvo como: {nome_arquivo}")

def gera_qr_code():
    while True:
        url = input("Digite a URL: ")
        if validators.url(url):
            break
    img = qrcode.make(url)
    img.save("qrcode.png")
    print("QrCode gerado com sucesso")


def main():
    result = direcinamento()
    if result == 1:
        gera_cod_barra()
    elif result == 2:
        gera_qr_code()
    else:
        print("Erro")

if __name__ == "__main__":
    main()