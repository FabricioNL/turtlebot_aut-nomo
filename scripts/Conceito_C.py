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
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import Image, CompressedImage, LaserScan
from cv_bridge import CvBridge, CvBridgeError
import cv2.aruco as aruco
import sys


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

#-- Font for the text in the image
font = cv2.FONT_HERSHEY_PLAIN

id = None
cimg = []
x_medio = 0
MENOR15M = False 
ACHOU150 = False
ACHOU50 = False
contador = 0

areaAzul = 0
areaVerde = 0
areaLaranja = 0


centro_verde = 0
centro_azul = 0
centro_laranja = 0

MISSAO = 'LARANJA'

def ajuste_linear_x_fy(mask):
    """Recebe uma imagem já limiarizada e faz um ajuste linear
        retorna coeficientes linear e angular da reta
        e equação é da forma
        y = coef_angular*x + coef_linear
    """ 
    pontos = np.where(mask==255)
    if len(pontos[0]) != 0:
        ximg = pontos[1]
        yimg = pontos[0]
        yimg_c = sm.add_constant(yimg)
        model = sm.OLS(ximg,yimg_c)
        results = model.fit()
        coef_angular = results.params[1] # Pegamos o beta 1
        coef_linear =  results.params[0] # Pegamso o beta 0
        return coef_angular, coef_linear
    return None, None


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
    if coef_angular != None:
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
    return None, None, None
    
def corta_imagem(mask):
    x_medio = int(mask.shape[1]/2)
    
    img_e = mask[0:mask.shape[0], 0: x_medio + 50]
    img_d = mask[0:mask.shape[0], x_medio+100: mask.shape[1]]
    
    return img_e, img_d


def calcular_angulo_com_vertical(img, lm):
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

def maior_contorno(contornos):
    maior_cont = None

    area = 0
    for contorno in contornos:
        areaC = cv2.contourArea(contorno)
        if areaC > area:
            maior_cont = contorno
            area = areaC
    
    return maior_cont

def area_contornos(contornos):
    area = 0

    for c in contornos:
        areaC = cv2.contourArea(c)
        
        area += areaC

    return area

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

bridge = CvBridge()

def center_of_contour(contorno):
    """ Retorna uma tupla (cx, cy) que desenha o centro do contorno"""
    M = cv2.moments(contorno)
    # Usando a expressão do centróide definida em: https://en.wikipedia.org/wiki/Image_moment
    if M["m00"] == 0:
        M["m00"] = 1
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return (int(cX), int(cY))

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
    leituras = np.array(dado.ranges).round(decimals=2)
    #print("Faixa valida: ", dado.range_min , " - ", dado.range_max )
    #print("Leituras:")
    ranges = np.array(dado.ranges).round(decimals=2)
    minv = dado.range_min 
    maxv = dado.range_max
    #print(leituras[0])

    if leituras[0] < 1:
        MENOR15M = True
        #print(MENOR15M)
    
    else:
        MENOR15M = False
 
# A função a seguir é chamada sempre que chega um novo frame
def roda_todo_frame(imagem):
    #print("frame")
    try:
        global cimg
        global x_medio
        global ACHOU150
        global ACHOU50

        global centro_azul
        global centro_verde
        global centro_laranja

        global areaAzul 
        global areaVerde 
        global areaLaranja 


        #cria copias da imagem em hsv e BGR
        cv_image = bridge.compressed_imgmsg_to_cv2(imagem, "bgr8")
        hsv = cv2.cvtColor(cv_image.copy(), cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        #realiza a leitura das IDs
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        #localiza as IDs e muda as tags
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

        #obtem as coordenadas do centro da tela
        cimg = (int(cv_image.shape[1]/2), int(cv_image.shape[0]/2))
        #cria uma copia de imagem para contornos
        img_cont = cv_image.copy()

        #cria uma mascara para filtrar a linha amarela
        mask = segmenta_linha_amarela(hsv)
        
        #segmenta os creepers
        mask_azul = segmenta_creeper_azul(hsv.copy())
        mask_laranja = segmenta_creeper_laranja(cv_image.copy())
        mask_verde = segmenta_creeper_verde(cv_image.copy())

        contornos_azul = encontrar_contornos(mask_azul)
        contornos_laranja = encontrar_contornos(mask_laranja)
        contornos_verde = encontrar_contornos(mask_verde)

        #temos que pegar o maior contorno de cada cor, agora

        maior_cont_azul = maior_contorno(contornos_azul)
        maior_cont_laranja = maior_contorno(contornos_laranja)
        maior_cont_verde = maior_contorno(contornos_verde)

        #vamos salvar as areas
        if maior_cont_azul is not None:
            areaAzul = cv2.contourArea(maior_cont_azul)
        else:
            areaAzul = 0
        
        if maior_cont_verde is not None:
            areaVerde = cv2.contourArea(maior_cont_verde)
        else:
            areaVerde = 0
        if maior_cont_laranja is not None:
            areaLaranja = cv2.contourArea(maior_cont_laranja)
        else:
            areaLaranja = 0

        #agora temos que pegar o centro desses contornos de creeper

        centro_azul = center_of_contour(maior_cont_azul)
        centro_verde = center_of_contour(maior_cont_verde)
        centro_laranja = center_of_contour(maior_cont_laranja)

        #corta a tela em duas
        img_e, img_d = corta_imagem(mask)

        #descomentar linhas na hora do teste
        mask_regression, x_f, y_f = ajuste_linear_grafico_x_fy(img_e)
        x_medio = x_f

        #contornos da mascara amarela
        contornos = encontrar_contornos(mask)


        areaTotal = area_contornos(contornos)
        contornos_filtrados = area_contornos(contornos)

        img_cont = cv_image.copy()
        cv2.drawContours(img_cont, contornos, -1, [255, 0, 255], 3)

        #desenha na imagem
        cv2.circle(img_cont, cimg, 10, (0,255, 0), -1)
        

        #cv2.imshow("Camera", cv_image)
        #cv2.putText(img_cont, "Area total dos contornos: " + str(areaTotal), (50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        #cv2.putText(img_cont, "Numero de contornos: " + str(len(contornos)), (50,100), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        
        if centro_azul != 0:
            cv2.circle(img_cont, centro_azul, 6, (255,0, 255), -1)

        if centro_verde != 0:
            cv2.circle(img_cont, centro_verde, 6, (0,255, 255), -1)

        if centro_laranja != 0:
            cv2.circle(img_cont, centro_laranja, 6, (255,255, 0), -1)

        cv2.imshow("Contornos", img_cont)
        
        #cv2.imshow("Mascara", mask)
        if mask_regression is not None:
            cv2.circle(mask_regression, (x_f, y_f), 10, (255,0, 0), -1)
            cv2.circle(img_cont, (x_f,y_f), 10, (0,255, 0), -1)
            cv2.imshow("Mascara do prof", mask_regression)




        cv2.waitKey(1)
    except CvBridgeError as e:
        print('ex', e)

if __name__=="__main__":

    rospy.init_node("q3")

    topico_imagem = "/camera/image/compressed"
    velocidade_saida = rospy.Publisher("/cmd_vel", Twist, queue_size = 3 )
    recebe_scan = rospy.Subscriber("/scan", LaserScan, scaneou)
    recebedor = rospy.Subscriber(topico_imagem, CompressedImage, roda_todo_frame, queue_size=4, buff_size = 2**24)

    while not rospy.is_shutdown():
        vel = Twist(Vector3(0,0,0), Vector3(0,0,-0.2))
        #vel = Twist(Vector3(0,0,0), Vector3(0,0,0))

       
        if MISSAO == 'LARANJA' and (areaLaranja >= 500):
            if (centro_laranja[0] != 0) and (len(cimg) != 0):
                print("CENTRALIZANDO NO CREEPER LARANJA")
                print(centro_laranja)
                if (cimg[0] <= centro_laranja[0] + 10) and (cimg[0] >= centro_laranja[0] -10):
                  
                    vel = Twist(Vector3(0.2,0,0), Vector3(0,0,0))
                  
                else:
                    
                    if (cimg[0] > centro_laranja[0]):
                        vel = Twist(Vector3(0.2,0,0), Vector3(0,0,0.1))
                        #pass
                    if (cimg[0] < centro_laranja[0]):
                        vel = Twist(Vector3(0.2,0,0), Vector3(0,0,-0.1))
                        #pass
        
        elif MISSAO == 'AZUL' and (areaAzul >= 500):
            if (centro_azul != 0) and (len(cimg) != 0):
                print("CENTRALIZANDO NO CREEPER AZUL")
                print(centro_azul)
                if (cimg[0] <= centro_azul[0] + 10) and (cimg[0] >= centro_azul[0] -10):
                    vel = Twist(Vector3(0.1,0,0), Vector3(0,0,0))
                else:
                   
                    if (cimg[0] > centro_azul[0]):
                        vel = Twist(Vector3(0.1,0,0), Vector3(0,0,0.1))
                        #pass
                    if (cimg[0] < centro_azul[0]):
                       vel = Twist(Vector3(0.1,0,0), Vector3(0,0,-0.1))
                       #pass
        
        elif MISSAO == 'VERDE' and (areaVerde >= 500):
            if (centro_verde != 0) and (len(cimg) != 0):
                print("CENTRALIZANDO NO CREEPER VERDE")
                print(centro_verde)
                if (cimg[0] <= centro_verde[0] + 10) and (cimg[0] >= centro_verde[0] -10):
                    vel = Twist(Vector3(0.1,0,0), Vector3(0,0,0))
                    
                else:
                   
                    if (cimg[0] > centro_verde[0]):
                        vel = Twist(Vector3(0.1,0,0), Vector3(0,0,0.1))
                        #pass
                    if (cimg[0] < centro_verde[0]):
                       vel = Twist(Vector3(0.1,0,0), Vector3(0,0,-0.1))
                       #pass
        else:
            print('SEGUINDO PISTA AMARELA')
            if (MENOR15M and ACHOU150) or (MENOR15M and ACHOU50):
                    #print('GIRANDO')
                    vel = Twist(Vector3(0,0,0), Vector3(0,0,-0.3))
                    velocidade_saida.publish(vel)
                    rospy.sleep(6)
            else:
                if (len(cimg) != 0) and (x_medio != 0 and x_medio != None):

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

                
        rospy.sleep(0.1)
        velocidade_saida.publish(vel)