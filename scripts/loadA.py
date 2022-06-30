#! /usr/bin/env python3
# -*- coding:utf-8 -*-

from __future__ import print_function, division


# Para rodar, recomendamos que faça:
# 
#    roslaunch my_simulation pista_u.launch
#
# Depois o controlador do braço:
#
#    roslaunch mybot_description mybot_control2.launch 	
import rospy 
import statsmodels.api as sm
import numpy as np
import cv2
import time
import math
from geometry_msgs.msg import Twist, Vector3
from std_msgs.msg import Float64
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import Image, CompressedImage, LaserScan
from cv_bridge import CvBridge, CvBridgeError
import cv2.aruco as aruco
import sys
import os
import rospkg

Missao = ("orange",11,"cow")
#--- Define Tag de teste
id_to_find  = 200
marker_size  = 20 #- [cm]
#id_to_find  = 22
#marker_size  = 3 #- [cm]
# 

#--- Define the aruco dictionary
aruco_dict  = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters  = aruco.DetectorParameters_create()
parameters.minDistanceToBorder = 0
parameters.adaptiveThreshWinSizeMax = 1000

rospack = rospkg.RosPack()
path = rospack.get_path('ros_projeto')
scripts = os.path.join(path,  "scripts")
proto = os.path.join(scripts,"MobileNetSSD_deploy.prototxt.txt")
model = os.path.join(scripts, "MobileNetSSD_deploy.caffemodel")
confianca = 0.2

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]


COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
# print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(proto, model)

#-- Font for the text in the image
font = cv2.FONT_HERSHEY_PLAIN

id = None
cimg = []
x_medio = 0
X_aruco = 1
Y_aruco = 1
x_estacao = 1
mask_querida_id = 1
id_querido = 1
estacao_querida = 1
MENOR15M = False 
ACHOU150 = False
ACHOU50 = False
contador = 0
LinhaAmarela = True
ProcuraCreeper = False
AchouCreeper = False
PreVoltaPista = False
VoltaPista = False
VoltaPista = False
Creeper = False
IdAchado = False
ESTACAO = False
Inicio = True
PARADA = False



def detect(frame):
    """
        funcao que roda a rede neural e devolve uma imagem com o que foi detectado como também uma lista do que foi 
        detecado e as sua certeza
    """

    image = frame.copy()
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

    # pass the blob through the network and obtain the detections and
    # predictions
    # print("[INFO] computing object detections...")
    net.setInput(blob)
    detections = net.forward()

    results = []

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence


        if confidence > confianca:
            # extract the index of the class label from the `detections`,
            # then compute the (x, y)-coordinates of the bounding box for
            # the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # display the prediction
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            #print("[INFO] {}".format(label))
            cv2.rectangle(image, (startX, startY), (endX, endY),
                COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(image, label, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

            results.append((CLASSES[idx], confidence*100, (startX, startY),(endX, endY) ))

    # show the output image
    return image, results

def ajuste_linear_x_fy(mask):
    """Recebe uma imagem já limiarizada e faz um ajuste linear
        retorna coeficientes linear e angular da reta
        e equação é da forma
        y = coef_angular*x + coef_linear
    """ 
    pontos = np.where(mask==255)
    ximg = pontos[1]
    yimg = pontos[0]
    yimg_c = sm.add_constant(yimg)
    model = sm.OLS(ximg,yimg_c)
    results = model.fit()
    coef_angular = results.params[1] # Pegamos o beta 1
    coef_linear =  results.params[0] # Pegamso o beta 0
    return coef_angular, coef_linear

def ajuste_linear(mask):
    """Recebe uma imagem já limiarizada e faz um ajuste linear
        retorna coeficientes linear e angular da reta
        e equação é da forma
        y = coef_angular*x + coef_linear
    """ 
    pontos = np.where(mask==255)
    ximg = pontos[1]
    yimg = pontos[0]
    yimg_c = sm.add_constant(yimg)
    ximg_c = sm.add_constant(ximg)
    model = sm.OLS(yimg,ximg_c)
    results = model.fit()
    coef_angular = results.params[1] # Pegamos o beta 1
    coef_linear =  results.params[0] # Pegamso o beta 0
    return coef_angular, coef_linear

def segmenta_creeper_laranja(bgr):
    """Não mude ou renomeie esta função
        deve receber uma imagem bgr e retornar os segmentos amarelos do centro da pista em branco.
        Utiliza a função cv2.morphologyEx() para limpar ruidos na imagem
    """
    
    #vamos segmentar pelo intervalo do HSV. É o melhor jeito de segmentar a partir de tons coloridos/compostos
    #A ideia é que apenas os tons amarelos sejam brancos e todo o restante da imagem fique preto
    
    bgr2 = bgr.copy()
    hsv = cv2.cvtColor(bgr2, cv2.COLOR_BGR2HSV)
    
    #antes o hsv1 era 11 mas ajustando pra 21 ficou melhor. Esse intervalo foi obtido pelo color picker
    cor_hsv1  = np.array([ 0, 200, 190], dtype=np.uint8)
    cor_hsv2 = np.array([ 15, 255, 255], dtype=np.uint8)
    #cria uma mascara com base nos valores da cor1 e cor2
    mascara_amarela = cv2.inRange(hsv, cor_hsv1, cor_hsv2)
    
    mascara_amarela_blur = cv2.blur(mascara_amarela, (1,1))

    mask_morpho = morpho_limpa(mascara_amarela_blur)
    
    #plt.imshow(mascara_amarela_blur, cmap = "Greys_r")
    return mask_morpho

def segmenta_creeper_verde(bgr):
    """Não mude ou renomeie esta função
        deve receber uma imagem bgr e retornar os segmentos amarelos do centro da pista em branco.
        Utiliza a função cv2.morphologyEx() para limpar ruidos na imagem
    """
    
    #vamos segmentar pelo intervalo do HSV. É o melhor jeito de segmentar a partir de tons coloridos/compostos
    #A ideia é que apenas os tons amarelos sejam brancos e todo o restante da imagem fique preto
    
    bgr2 = bgr.copy()
    hsv = cv2.cvtColor(bgr2, cv2.COLOR_BGR2HSV)
    
    #antes o hsv1 era 11 mas ajustando pra 21 ficou melhor. Esse intervalo foi obtido pelo color picker
    cor_hsv1  = np.array([ 40, 50, 50], dtype=np.uint8)
    cor_hsv2 = np.array([ 70, 255, 255], dtype=np.uint8)
    #cria uma mascara com base nos valores da cor1 e cor2
    mascara_amarela = cv2.inRange(hsv, cor_hsv1, cor_hsv2)
    
    mascara_amarela_blur = cv2.blur(mascara_amarela, (1,1))

    mask_morpho = morpho_limpa(mascara_amarela_blur)
    
    #plt.imshow(mascara_amarela_blur, cmap = "Greys_r")
    return mask_morpho

def segmenta_creeper_azul(bgr):
    """Não mude ou renomeie esta função
        deve receber uma imagem bgr e retornar os segmentos amarelos do centro da pista em branco.
        Utiliza a função cv2.morphologyEx() para limpar ruidos na imagem
    """
    
    #vamos segmentar pelo intervalo do HSV. É o melhor jeito de segmentar a partir de tons coloridos/compostos
    #A ideia é que apenas os tons amarelos sejam brancos e todo o restante da imagem fique preto
    
    bgr2 = bgr.copy()
    hsv = cv2.cvtColor(bgr2, cv2.COLOR_BGR2HSV)
    
    #antes o hsv1 era 11 mas ajustando pra 21 ficou melhor. Esse intervalo foi obtido pelo color picker
    cor_hsv1  = np.array([ 0, 50, 50], dtype=np.uint8)
    cor_hsv2 = np.array([ 15, 170, 255], dtype=np.uint8)
    #cria uma mascara com base nos valores da cor1 e cor2
    mascara_amarela = cv2.inRange(hsv, cor_hsv1, cor_hsv2)
    
    mascara_amarela_blur = cv2.blur(mascara_amarela, (1,1))

    mask_morpho = morpho_limpa(mascara_amarela_blur)
    
    #plt.imshow(mascara_amarela_blur, cmap = "Greys_r")
    return mask_morpho

def ajuste_linear_grafico_x_fy(mask):
    """Faz um ajuste linear e devolve uma imagem rgb com aquele ajuste desenhado sobre uma imagem"""
    coef_angular, coef_linear = ajuste_linear_x_fy(mask)
    #print("x = {:3f}*y + {:3f}".format(coef_angular, coef_linear))
    pontos = np.where(mask==255) # esta linha é pesada e ficou redundante
    ximg = pontos[1]
    yimg = pontos[0]
    y_bounds = np.array([min(yimg), max(yimg)])
    x_bounds = coef_angular*y_bounds + coef_linear
    #print("x bounds", x_bounds)
    #print("y bounds", y_bounds)
    x_int = x_bounds.astype(dtype=np.int64)
    y_int = y_bounds.astype(dtype=np.int64)
    mask_rgb =  cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    x_f = x_int[0]
    y_f = y_int[0]
    cv2.line(mask_rgb, (x_int[0], y_int[0]), (x_int[1], y_int[1]), color=(0,0,255), thickness=11);    
    return mask_rgb, x_f, y_f 
    
def corta_imagem(mask):
    """
        funcao que corta a imagem em lado direito e esquerdo e devolve as duas
    """

    x_medio = int(mask.shape[1]/2)
    
    img_e = mask[0:mask.shape[0], 0: x_medio + 50]
    img_d = mask[0:mask.shape[0], x_medio+100: mask.shape[1]]
    
    return img_e, img_d


def calcular_angulo_com_vertical(img, lm):
    """
        funcao que calcula o angulo formado entre a regressao e a vertical. Devolve o angulo em graus
    """
    
    x_0 = lm[0][0]
    y_0 = lm[0][1]
    
    x_1 = lm[1][0]
    y_1 = lm[1][0]
    
    vetor_horizontal = np.array([0,100])
    vetor_reta = np.array([x_1 - x_0, y_1 - y_0])
    
    #agora que temos dois vetores vamos realizar o produto vetorial dividido pelo módulo
    modulo_horizontal = (vetor_horizontal[0]**2 + vetor_horizontal[1]**2)**0.5
    modulo_reta = (vetor_reta[0]**2 + vetor_reta[1]**2)**0.5
    
    produto_escalar = (vetor_horizontal[0]*vetor_reta[0] + vetor_horizontal[1]*vetor_reta[1])
    
    angulo_radiano = math.acos((produto_escalar)/(modulo_horizontal*modulo_reta))
    
    angulo_graus = math.degrees(angulo_radiano)
    
    return angulo_graus

def segmenta_linha_amarela(bgr):
    """Não mude ou renomeie esta função
        deve receber uma imagem bgr e retornar os segmentos amarelos do centro da pista em branco.
        Utiliza a função cv2.morphologyEx() para limpar ruidos na imagem
    """
    
    #vamos segmentar pelo intervalo do HSV. É o melhor jeito de segmentar a partir de tons coloridos/compostos
    #A ideia é que apenas os tons amarelos sejam brancos e todo o restante da imagem fique preto
    
    bgr2 = bgr.copy()
    hsv = cv2.cvtColor(bgr2, cv2.COLOR_BGR2HSV)
    
    #antes o hsv1 era 11 mas ajustando pra 21 ficou melhor. Esse intervalo foi obtido pelo color picker
    cor_hsv1  = np.array([ 27, 200, 249], dtype=np.uint8)
    cor_hsv2 = np.array([ 32, 230, 255], dtype=np.uint8)
    #cria uma mascara com base nos valores da cor1 e cor2
    mascara_amarela = cv2.inRange(hsv, cor_hsv1, cor_hsv2)
    
    mascara_amarela_blur = cv2.blur(mascara_amarela, (1,1))

    mask_morpho = morpho_limpa(mascara_amarela_blur)
    
    #plt.imshow(mascara_amarela_blur, cmap = "Greys_r")
    return mask_morpho

def encontrar_contornos(mask):
    """Não mude ou renomeie esta função
        deve receber uma imagem preta e branca os contornos encontrados
    """
    
    #ele vai contornar o que está em branco
    
    contornos, arvore = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
   
    return contornos

def area_contornos(contornos):
    """
        Funcao que percorre os contornos e soma suas áreas, devolvendo a area total
    """

    area = 0

    for c in contornos:
        areaC = cv2.contourArea(c)
        
        area += areaC

    return area



def centro_aruco(corner):
    """
        Funcao que pega as coordenadas do aruco e devolve seu centro como (x,y)
    """

    Xa = int((corner[0][1][0]-corner[0][0][0])/2 + corner[0][0][0])
    Ya = int((corner[0][2][1]-corner[0][1][1])/2 + corner[0][1][1])
    return (Xa,Ya)
    
    
def distanciaEuclidiana(Po,Pf):
    """
        Funcao que calcula a distancia entre dois pontos e devolve
    """

    Xo = Po[0]
    Yo = Po[1]

    Xf = Pf[0]
    Yf = Pf[1]

    dis = ((Xf-Xo)**2+(Yf-Yo)**2)**(1/2)
    
    return dis
    


def encontrar_centro_dos_contornos(img, contornos):
    """Não mude ou renomeie esta função
        deve receber um contorno e retornar, respectivamente, a imagem com uma cruz no centro de cada segmento e o centro dele. formato: img, x, y
    """
    lista_x = []
    lista_y = []
    
    for i in contornos:
        M = cv2.moments(i)
        # Usando a expressão do centróide definida em: https://en.wikipedia.org/wiki/Image_moment
        if M["m00"] == 0: 
            M["m00"] = 1
        
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        crosshair(img, (cX,cY), 10, (255,255,0))
        
        #isso serve pra filtrar e garantir que são pontos presentes na imagem
        if (cX and cY) != 0:
        
            lista_x.append(cX)
            lista_y.append(cY)
        
    return img, lista_x, lista_y

def morpho_limpa(mask):
    """
        Funcao que limpa os defeitos da mascara por meio do morph e retorna a máscara mais limpa
    """

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    mask = cv2.morphologyEx( mask, cv2.MORPH_OPEN, kernel )
    mask = cv2.morphologyEx( mask, cv2.MORPH_CLOSE, kernel )    
    return mask

x_medio = 0
ranges = None
minv = 0
maxv = 10
cimg = [0,0]
ANGULO = 0
XCreeper = 1
YCreeper = 1

bridge = CvBridge()

def crosshair(img, point, size, color):
    """ Desenha um crosshair centrado no point.
        point deve ser uma tupla (x,y)
        color é uma tupla R,G,B uint8
    """
    #o crosshair vai desenhar um centro a partir dos centros encontrados na formula de centro dos contornos
    
    x,y = point
    cv2.line(img,(x - size,y),(x + size,y),color,2)
    cv2.line(img,(x,y - size),(x, y + size),color,2)


def regressao_por_centro(img, X,Y):
    """Não mude ou renomeie esta função
        deve receber uma lista de coordenadas XY, e estimar a melhor reta, utilizando o metodo preferir, que passa pelos centros. Retorne a imagem com a reta e os parametros da reta
        
        Dica: cv2.line(img,ponto1,ponto2,color,2) desenha uma linha que passe entre os pontos, mesmo que ponto1 e ponto2 não pertençam a imagem.
    """
    
    #as coordenadas x e y são as coordenadas dos contornos. É por eles que vai ser possível criar a regressão.
    
    #transforma as listas em vetores
    x = np.array(X)
    y = np.array(Y)
    
    #adiciona uma constante
    y_c = sm.add_constant(y)
    #define o modelo
    model = sm.OLS(x,y_c)
    #encontra os resultados
    results = model.fit()
    
    coef_angular = results.params[1]
    coef_linear = results.params[0]
    
    y_min = 0
    y_max = img.shape[0]
    
    x_min = int(coef_angular*y_min + coef_linear)
    x_max = int(coef_angular*y_max + coef_linear)

    cv2.line(img, (x_min, y_min), (x_max, y_max), (255,255,0), thickness=3);       
    
    return img, [(x_min, y_min),(x_max,y_max)]

def scaneou(dado):
    global ranges
    global minv
    global maxv
    global MENOR15M 
    global AchouCreeper
    global VoltaPista
    global ProcuraCreeper
    global PreVoltaPista
    global ESTACAO
    global PARADA

    leituras = np.array(dado.ranges).round(decimals=3)
    #print("Faixa valida: ", dado.range_min , " - ", dado.range_max )
    #print("Leituras:")
    ranges = np.array(dado.ranges).round(decimals=2)
    minv = 0.1 
    maxv = 10
    #print(leituras[0])
    distancia = 0.215
    distancia2 = 1

    if leituras[0] < 1:
        MENOR15M = True
        #print(MENOR15M)
    
    else:
        MENOR15M = False

    if (leituras[0] <= distancia or leituras[359] <= distancia or leituras[2] <= distancia or leituras[350] <= distancia or leituras[4] <= distancia or leituras[5] <= distancia)  and ProcuraCreeper == True:
        PreVoltaPista = True
        AchouCreeper = False
        ProcuraCreeper = False
    
    if (leituras[0] <= distancia2 or leituras[359] <= distancia2 or leituras[2] <= distancia2 or leituras[350] <= distancia2 or leituras[4] <= distancia2 or leituras[5] <= distancia2) and ESTACAO:
        ESTACAO = False
        PARADA = True

 
# A função a seguir é chamada sempre que chega um novo frame
def roda_todo_frame(imagem):
    #print("frame")
    try:
        global cimg
        global x_medio
        global ACHOU150
        global ACHOU50
        global ProcuraCreeper
        global XCreeper
        global YCreeper
        global PreVoltaPista
        global LinhaAmarela
        global Creeper
        global X_aruco
        global Y_aruco
        global IdAchado
        global VoltaPista
        global mask_querida_id
        global estacao_querida
        global Missao
        global x_estacao
        global ESTACAO

        cv_image = bridge.compressed_imgmsg_to_cv2(imagem, "bgr8")
        hsv = cv2.cvtColor(cv_image.copy(), cv2.COLOR_BGR2HSV)

        img_cont = cv_image.copy()

        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        #print(ids)

        # Estações
        image_net,estacoes = detect(cv_image.copy())

        for estacao in estacoes:
            if estacao[0] == estacao_querida and estacao[1] > 70  and Creeper == False: #LinhaAmarela == True
                x_estacao = int((estacao[2][0]+estacao[3][0])/2)
                y_estacao = int((estacao[2][1]+estacao[3][1])/2)
                ESTACAO = True
                LinhaAmarela = False
                print(x_estacao, y_estacao)
                cv2.circle(image_net (x_estacao,y_estacao), 10, (255,255, 0), -1)

        #ID que queremos procurar
        #id_querido = 12

        if ids is not None:
            for i_d in ids:
                if i_d== [150]:
                    ACHOU150 = True
                else:
                    ACHOU150 = False

                if i_d == [50]:
                    ACHOU50 = True
                else:
                    ACHOU50 = False
            t = 0
            #Encontra o ID correto e marca o centro do ID
            for i_d in ids:
                if i_d == [id_querido]:
                    corner = corners[t]
                    CentroAruco = centro_aruco(corner)
                    X_aruco = CentroAruco[0]
                    Y_aruco = CentroAruco[1]
                    cv2.circle(img_cont,CentroAruco , 5, (255,0, 255), -1)
                    IdAchado = True
                    break
                else:
                    t += 1
                    IdAchado = False

        #print(X_aruco,Y_aruco)

        cimg = (int(cv_image.shape[1]/2), int(cv_image.shape[0]/2))
        
        

        mask = segmenta_linha_amarela(hsv)
        
        mask_azul = segmenta_creeper_azul(hsv.copy())
        mask_laranja = segmenta_creeper_laranja(cv_image.copy())
        mask_verde = segmenta_creeper_verde(cv_image.copy())

        if mask_querida_id == "blue" :
            mask_querida = mask_azul.copy()

        elif mask_querida_id == "green" :
            mask_querida = mask_verde.copy()

        elif mask_querida_id == "orange" :
            mask_querida = mask_laranja.copy()

        else:
            mask_querida = mask_azul.copy()

        #Código para encontrar o Creeper
        if Creeper:
            ContornoCreeper = encontrar_contornos(mask_querida)
            
            if len(ContornoCreeper) != 0 and VoltaPista == False and PreVoltaPista == False and IdAchado == True:

            
                ImgCentroCreeper = mask_querida.copy()
                ImgCentroCreeper,XCreeperl,YCreeperl = encontrar_centro_dos_contornos(ImgCentroCreeper,ContornoCreeper)
                cv2.imshow("Creeper", ImgCentroCreeper)
                XCreeper = XCreeperl[0]
                YCreeper = YCreeperl[0]
                CentroCreeper = (XCreeper, YCreeper)

                if distanciaEuclidiana(CentroAruco,CentroCreeper) < 50 :
                    ProcuraCreeper = True
                    LinhaAmarela = False
                    XCreeper = X_aruco
                    YCreeper = Y_aruco
                


        

        img_e, img_d = corta_imagem(mask)
        contornos = encontrar_contornos(mask)

        #descomentar linhas na hora do teste
        if len(contornos) != 0 :
            mask_regression, x_f, y_f = ajuste_linear_grafico_x_fy(img_e)
            x_medio = x_f

        

        if len(contornos) > 6 and VoltaPista == True:
            VoltaPista = False
            LinhaAmarela = True
            Creeper = False

        areaTotal = area_contornos(contornos)
        contornos_filtrados = area_contornos(contornos)

        #img_cont = cv_image.copy()
        cv2.drawContours(img_cont, contornos, -1, [255, 0, 255], 3)


        cv2.circle(img_cont, cimg, 10, (0,255, 0), -1)
        if len(contornos) != 0 :
            cv2.circle(img_cont, (x_f,y_f), 10, (0,255, 0), -1)

        #cv2.imshow("Camera", cv_image)
        #cv2.putText(img_cont, "Area total dos contornos: " + str(areaTotal), (50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        #cv2.putText(img_cont, "Numero de contornos: " + str(len(contornos)), (50,100), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        
        cv2.imshow("Contornos", img_cont)
        
        #cv2.imshow("Mascara", mask)
        if len(contornos) != 0 :
            cv2.circle(mask_regression, (x_f, y_f), 10, (255,0, 0), -1)
            cv2.imshow("Mascara do prof", mask_regression)
            cv2.imshow("net", image_net)




        cv2.waitKey(1)
    except CvBridgeError as e:
        print('ex', e)

if __name__=="__main__":

    rospy.init_node("q3")

    topico_imagem = "/camera/image/compressed"
    velocidade_saida = rospy.Publisher("/cmd_vel", Twist, queue_size = 3 )
    recebe_scan = rospy.Subscriber("/scan", LaserScan, scaneou)
    recebedor = rospy.Subscriber(topico_imagem, CompressedImage, roda_todo_frame, queue_size=4, buff_size = 2**24)
    ombro = rospy.Publisher("/joint1_position_controller/command", Float64, queue_size=1)
    garra = rospy.Publisher("/joint2_position_controller/command", Float64, queue_size=1)
    l = 0
    while not rospy.is_shutdown():
        #vel = Twist(Vector3(0,0,0), Vector3(0,0,0.2))
        vel = Twist(Vector3(0,0,0), Vector3(0,0,0))
        if Inicio:
            print("Inicio")   
            ombro.publish(-0.36)
            garra.publish(-1.0)
            if l > 10:
                Creeper = True
                Inicio = False
            if l == 8:
                mask_querida_id = str(Missao[0])
                id_querido = int(Missao[1])
                estacao_querida = str(Missao[2])
                l += 1
            else:
                l += 1
        elif LinhaAmarela:
            print("LinhaAmarela")
            if (MENOR15M and ACHOU150) or (MENOR15M and ACHOU50):
                    #print('GIRANDO')
                    vel = Twist(Vector3(0,0,0), Vector3(0,0,0.3))
                    velocidade_saida.publish(vel)
                    rospy.sleep(11.2)
            else:
                if (len(cimg) != 0) and (x_medio != 0):

                        #print(cimg)
                        #print(x_medio)

                    if (cimg[0] <= x_medio + 10) and (cimg[0] >= x_medio -10):
                        ##print(cimg)
                        #print(x_medio)
                        #print('CENTRALIZEI')
                        vel = Twist(Vector3(0.2,0,0), Vector3(0,0,0))
                        #pass

                    else:
                        #print(cimg)
                        #print(x_medio)
                        #print('girando')
                        if (cimg[0] > x_medio):
                            vel = Twist(Vector3(0.15,0,0), Vector3(0,0,0.2))
                            #pass
                        if (cimg[0] < x_medio):
                            vel = Twist(Vector3(0.15,0,0), Vector3(0,0,-0.2))
                            #pass
        
        elif ProcuraCreeper:
            print("ProcuraCreeper")
            if (cimg[0] <= XCreeper + 2) and (cimg[0] >= XCreeper - 2):
                vel = Twist(Vector3(0.2,0,0), Vector3(0,0,0))
                AchouCreeper= True


            elif (cimg[0] > XCreeper):
                vel = Twist(Vector3(0.2,0,0), Vector3(0,0,0.15))

            elif (cimg[0] < XCreeper):
                vel = Twist(Vector3(0.2,0,0), Vector3(0,0,-0.15))
        
        
        elif PreVoltaPista:
            print("PreVoltaPista")
            vel = Twist(Vector3(0,0,0), Vector3(0,0,0))
            velocidade_saida.publish(vel)
            #rospy.sleep(0.1)
            #ombro.publish(0.0)
            rospy.sleep(0.05)
            garra.publish(0.0)
            rospy.sleep(1.5)
            ombro.publish(4.0)
            VoltaPista = True
            PreVoltaPista = False
            Creeper = False

        
        elif VoltaPista:
            print("VoltaPista")
            vel = Twist(Vector3(0,0,0), Vector3(0,0,-0.2))

        elif ESTACAO:
            print("Estacao")
            if (cimg[0] <= x_estacao + 2) and (cimg[0] >= x_estacao - 2):
                vel = Twist(Vector3(0.2,0,0), Vector3(0,0,0))
                print("CENTRALIZADO")

            elif (cimg[0] > x_estacao):
                vel = Twist(Vector3(0.15,0,0), Vector3(0,0,0.1))
                print("GIRANDO")

            elif (cimg[0] < x_estacao):
                vel = Twist(Vector3(0.15,0,0), Vector3(0,0,-0.1))
                print("GIRANDO")
        elif PARADA:
            vel = Twist(Vector3(0,0,0), Vector3(0,0,0))
            velocidade_saida.publish(vel)
            ombro.publish(-1.0)
            garra.publish(-1.0)
                
        rospy.sleep(0.1)
        velocidade_saida.publish(vel)


