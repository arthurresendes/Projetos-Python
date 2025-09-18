import pyautogui
import time

"""
Verifique se o repositorio aberto é o que tem a devida atualização e verifique as coordenadas , pois variam de monitor para monitor
"""

def main():
    pyautogui.press("win")
    pyautogui.write("Github Desktop")
    pyautogui.press("enter")
    time.sleep(2)
    pyautogui.click(x=130,y=795)
    pyautogui.write("Atualizar")
    time.sleep(1)
    pyautogui.click(x=152,y=988)
    time.sleep(1)
    pyautogui.click(x=1374,y=357)


if __name__ == "__main__":
    main()