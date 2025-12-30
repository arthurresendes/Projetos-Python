from withoutbg import WithoutBG


img = WithoutBG.opensource()
resultado = img.remove_background("invencivel_com_fundo.jpg")
resultado.save("invencivel_sem_fundo.png")