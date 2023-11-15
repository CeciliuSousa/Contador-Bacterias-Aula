import os

import cv2
import numpy as np

# Diretório das imagens
diretorio_imagens = "imagens"

# Lista todos os arquivos no diretório
for arquivo in os.listdir(diretorio_imagens):
    if arquivo.endswith(".png") or arquivo.endswith(".jpg"):
        imagem_path = os.path.join(diretorio_imagens, arquivo)

        imagem_original = cv2.imread(imagem_path)
        imagem_original = cv2.resize(imagem_original, (300, 300))

        imagem = cv2.imread(imagem_path)
        imagem = cv2.resize(imagem, (300, 300))
        gray = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)

        imagem_colorida = cv2.imread(imagem_path)
        imagem_colorida = cv2.resize(imagem_colorida, (300, 300))

        # Aplicar suavização para reduzir ruído
        img = cv2.GaussianBlur(gray, (5, 5), 3)
        img = cv2.Canny(img, 40, 120)
        kernel = np.ones((1, 1), np.uint8)
        img = cv2.dilate(img, kernel, iterations=2)
        img = cv2.erode(img, kernel, iterations=2)

        # Encontrar os contornos
        contours, _ = cv2.findContours(
            img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Desenhar contornos na imagem original e adicionar números
        for i, cnt in enumerate(contours):
            cv2.drawContours(imagem_colorida, [cnt], -1, (0, 255, 0), 2)
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(imagem_colorida, str(
                    i+1), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 0, 0), 1, cv2.LINE_AA)

        # Contar o número de células
        print(f"No. de células na imagem {arquivo}: {len(contours)}")

        cv2.imshow("imagem com contornos e números", imagem_colorida)
        cv2.imshow("imagem pre-processada", img)
        cv2.imshow("imagem cinza", gray)
        cv2.imshow("imagem original", imagem_original)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
