# Detectskin

# Este programa, esta em fase de teste, e faz parte do processamento de vídeos para teste do tempo de enchimento capilar.  

# O objetivo é obter somente a região do dedo, no momento do teste.
Primeiramente fazemos uma detecção da pele usando a conversão do video (RGB) para HSV, após isso fazemos um threshold de cor e aplicamos operadores morfológicos de erosão e dilatação para construir uma máscara, aplica-se um filtro gaussiano para remover os ruidos e assim, corta-se o video no tamanho de máxima quantidade de frames que se interpreta ter a região da pele. 
Lembrar que deve-se escolher uma quantidade de frames iniciais, onde se tenha somente a região da pele antes do inicio do teste de CRT.
