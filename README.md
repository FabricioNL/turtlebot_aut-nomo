#Projeto 1 - Robótica 

Objetivo do projeto: 

O projeto consiste em um robô virtual em ROS capaz de simular mecanicamente um turtlebot waffle. Assim, através do software desenvolvido, o robô recebe uma missão de levar dos creepers identificado pela cor e pelo QR code até uma das bases temáticas.

Para fazer isso, ele realiza uma regressão linear dos pontos de centro de massa da pista amarela, de forma a percorrer a pista. Além disso, também aplica um filtro através do OpenCV e identifica o creeper da cor desejada e aplica um segundo check no QR code, para que seja também o código desejado. Após isso, ele volta para a pista e tenta encontrar a base designada. A base é identificada a partir de uma rede neural e caso a tag definida pela rede neural tenha mais que 80% de precisão, ele segue até ela.


Nome dos integrantes: 

* Fabricio Neri Lima
* Jean Silas Ferreira Sanandrez


**Videos do conceito C**

Os videos da entrega correspondente ao conceito C estão disponíveis abaixo:

[Robô andando na pista](https://www.youtube.com/watch?v=opD_GfTo9PM)

[Robô batendo no creeper laranja](https://www.youtube.com/watch?v=8h0dZiuNIvo&t=21s&ab_channel=Fabr%C3%ADcioNeriFabr%C3%ADcioNeri)

[Robô batendo no creeper azul](https://www.youtube.com/watch?v=83dlFYKz_SI&ab_channel=Fabr%C3%ADcioNeriFabr%C3%ADcioNeri) 

[Robô batendo no creeper verde](https://www.youtube.com/watch?v=cmz_YSiCJVs&ab_channel=Fabr%C3%ADcioNeriFabr%C3%ADcioNeri)

**Video do conceito B** 

Para realizar o conceito B, foi necessário girar os creepers presentes no círculo. Isso porque através da movimentação por meio da regressão,  o robô sempre anda para a esquerda, então ele não encontra o robô na posição inicial.

![creepers](https://user-images.githubusercontent.com/39420630/119387986-16e7a080-bca0-11eb-8475-9b85e8ee5080.png)


[Pegando o creeper azul](https://www.youtube.com/watch?v=fPWd4KOpTgI&ab_channel=JeanSilasFerreiraSanandrez) 

[Pegando o creeper verde](https://www.youtube.com/watch?v=-1osvyPIv2s&ab_channel=JeanSilasFerreiraSanandrez) 

[Pegando o creeper Laranja](https://www.youtube.com/watch?v=szYMWNQaFHE&ab_channel=JeanSilasFerreiraSanandrez)


**Video do Conceito A (Parcial)**

[Creeper Azul na Base DOG](https://www.youtube.com/watch?v=PuIyQQryiAc&ab_channel=JeanSilasFerreiraSanandrez) 

[Creeper Verde na Base HORSE](https://www.youtube.com/watch?v=OlGEWcGXRMM&t=1s) 

[Creeper Laranja na Base Cow](https://www.youtube.com/watch?v=nG4aP9L-Bwg&ab_channel=Fabr%C3%ADcioNeriFabr%C3%ADcioNeri) 
