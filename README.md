# uff.traffic.sign.recognition
Projeto de conclusão da disciplina de Análise de Imagem

UNIVERSIDADE FEDERAL FLUMINENSE - Programa de Pós-Graduação em Computação

Disciplina ministrada por:
Aura Conci

## The main goal is to find the traffic sign in a photo and highlight it

### Prerequisites

* Conhecimento de Python
* Conhecimento de análise de imagem
* Conhecimento de visão computacional

### Installing

* Instalar o Python na versão 3.7
* Instalar todos os pacotes contidos no arquivo requirements.txt

### Versão 2.0 - Agora usando segmentação por cores da fotografia

O processo consiste em:

### 1. Imagem de entrada

Obter imagens por uma câmera ligada ao carro que fica no interior do mesmo.

O processamento será realizado on-line no servidor da AWS e o sistema no interior do carro ficara responsável somente por envia as fotografias capturadas para a API descrita neste projeto.

A primeira ação na imagem será aplicar um filtro bilateral para suavização e para remover ruídos.

```
blur = cv2.bilateralFilter(imagemDeEntrada, 9, 75, 75)
```

### 2. Espaço de cor ideal

A segunda parte do processamento consiste em converter as imagens do espaço RGB (Red, Green, Blue) para outro espaço de cor "HSV".

HSV - Sistemas de cores formado pelos componentes:
- Matiz (hue). Representa todas as cores puras; Geralmente é representada por ângulos; Começa em 0 no vermelho e termina em 360 também no vermelho.
- Saturação (saturation). O quanto a cor possui o componente de cor branca.
- Brilho (value). Noção acromática de intensidade.

```
hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
```

### 3. Segmentação da imagem

Em sequencia, com um corte do canal da Matiz, se mantem as cores mais puras do Vermelho e Amarelo:
Um corte no canal de saturação para manter somente os valores altos;
E um corte no canal de iluminação também para manter os valores altos.

```
hsvWhite  = np.asarray([0, 0, 255])
hsvYellow = np.asarray([30, 255, 255])
mask = cv2.inRange(hsv, hsvWhite, hsvYellow)
interest = cv2.bitwise_and(hsv, hsv, mask=mask)
```

### 4. Binarização e limiarização da imagem

Para extrair contornos das formas que permaneceram na imagem e transformar nossa imagem em binária para futuros processos, aplicamos o filtro de Canny (pode ser outro mais simples, se o desempenho estiver ruim).
Com tamanho da abertura igual a 3.
It was developed by John F. Canny in 1986.

```
edges = cv2.Canny(interest, 50, 150, apertureSize = 3)
```

Depois para reduzir ou eliminar pequenos buracos dentro das formas/objetos encontrados, aplicamos dois filtros morfológicos: Dilatação (Dilation) seguido de erosão (Erosion).
Método Closing:

```
closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
```

### 5. Extração de features

Então aplicamos a transformada de Hough para detectar as linhas dos objetos na nossa imagem binária. Usaremos uma otimização do algoritmo proposta em 2008 por Leandro A.F. Fernandes (Professor de Visão Computacional na UFF), Manuel M. Oliveira

# Bibliografia:

- A. Conci, E. Azevedo e F.R. Leta, Computação Gráfica: Teoria e Prática, vol. 2., Elsevier, 2008

- Kernel-based Hough transform (KHT)
Fernandes, L.A.F.; Oliveira, M.M. (2008). "Real-time line detection through an improved Hough transform voting scheme". Pattern Recognition.