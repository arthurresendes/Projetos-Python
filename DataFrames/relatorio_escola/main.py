import pandas as pd
import matplotlib.pyplot as plt

def criando_dataframe():
    global df
    df = pd.DataFrame({
    'Aluno': ['Ana','Bruno','Carlos','Daniel','Eduard','Fernanda','Gabriel','Helena','Isabela','João'],
    'Curso': ['Engenharia','Direito','Engenharia','Medicina','Direito','Engenharia','Medicina','Direito','Engenharia','Medicina'],
    'Nota1': [9, 8, 7, 6, 8, 10, 9, 7, 8, 6],
    'Nota2': [8, 7, 6, 7, 9, 9, 8, 6, 7, 7],
    'Nota3': [10, 9, 7, 6, 8, 10, 9, 7, 8, 6]
    })

def relatorio():
    global relatorio_total
    relatorio_total = df.copy()
    cores = {'Engenharia':'blue', 'Direito':'red', 'Medicina':'green'}
    global colors
    colors = df['Curso'].map(cores)
    relatorio_total['Média Total'] = df[['Nota1','Nota2','Nota3']].mean(axis=1)
    relatorio_total['Máxima Total'] = df[['Nota1','Nota2','Nota3']].max(axis=1)
    relatorio_total['Mínima Total'] = df[['Nota1','Nota2','Nota3']].min(axis=1)
    relatorio_total['Desvio Padrão'] = df[['Nota1', 'Nota2', 'Nota3']].std(axis=1)

def contagem_condicoes_ordenacao():
    melhores_alunos = relatorio_total.sort_values("Média Total" ,ascending=False)
    print(f"Do melhor aluno para o pior:\n{melhores_alunos}\n")
    cursos_mais_acessados = relatorio_total['Curso'].value_counts()
    print(f"Cursos mais acessados:\n{cursos_mais_acessados}\n")
    reprovado = relatorio_total[relatorio_total['Média Total'] < 7]
    print(f"Alunos reprovados:\n{reprovado}")

def dash_media_curso():
    plt.figure(figsize=(8,4))
    plt.barh(relatorio_total["Curso"], relatorio_total["Média Total"], color=colors, alpha=0.6)
    plt.title("Média por curso")
    plt.xlabel("Médias")
    plt.ylabel("Curso")
    plt.show()

def dash_media_aluno():
    plt.figure(figsize=(8,4))
    plt.bar(relatorio_total["Aluno"], relatorio_total["Média Total"], color='blue', alpha=0.6)
    plt.title("Média de cada Aluno")
    plt.xlabel("Aluno")
    plt.ylabel("Médias")
    plt.show()

def main():
    criando_dataframe()
    relatorio()
    contagem_condicoes_ordenacao()
    dash_media_curso()
    dash_media_aluno()

if __name__ == "__main__":
    main()