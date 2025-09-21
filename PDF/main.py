import PyPDF2

'''
Para muitos arquivos na junção pode se utilizar isso , com os achando a pasta que estão os arquivos e juntando um por no loop for
import os

merger = PyPDF2.PdfMerger()

lista_arquivos = os.listdir("pasta")
for arq in lista_arquivos:
    if ".pdf" in arq:
        merger.append(f"pasta/{arq}")
merger.write("juncao.pdf")
merger.close()

'''

def main():
    merger = PyPDF2.PdfMerger()
    merger.append("exemplo1.pdf")
    merger.append("exemplo2.pdf")
    merger.write("juncao.pdf")
    merger.close()

if __name__ == "__main__":
    main()