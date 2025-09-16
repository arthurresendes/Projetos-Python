import pyautogui
import time

'''
Vale lembrar que as posições variam de computador para computador , sendo necessario usar um print(pyautogui.position())
'''

def indo_web():
    time.sleep(2)
    pyautogui.press('win')
    pyautogui.write("chrome")
    pyautogui.press("enter")
    time.sleep(5)
    pyautogui.click(x=631, y=662)

# Para garantir que o site vai para uma nova aba
def cria_aba():
    time.sleep(2)
    pyautogui.click(x=363, y=32)

def busca_site():
    pyautogui.write("https://arthurresendes.github.io/Portfolio/portfolio2.html")
    pyautogui.press('enter')

def especialidade():
    time.sleep(3)
    #print(pyautogui.position()) -> Para descobrir as coordenadas ficando com o mouse em cima
    pyautogui.click(x=695,y=274)

def main():
    indo_web()
    cria_aba()
    busca_site()
    especialidade()

if __name__ == "__main__":
    main()