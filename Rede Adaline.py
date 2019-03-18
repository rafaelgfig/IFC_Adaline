# -*- coding: utf-8 -*-
"""
Título: Uma rede neural muito simples - Adaline
Autor:  Rafael Goncalves
Data:   25/04/2018
Python: 3.6
"""

# Importação
import numpy as np
import matplotlib.pyplot as plt

# Taxa de aprendizado para o treino
taxaAprendizado = 0.3

# Seed dos números aleatórios para cálculos deterministicos
np.random.seed(5)

# Listas usadas para armazenar os erros e os pesos
erros=[]
bias =[]
peso1=[]
peso2=[]

# Função degrau
def step(x):
    if (x > 0):
        return 1
    return -1

# Primeiro exemplo. OR, linearmente separável
# Dados de entrada representando o operador logico OR (com o BIAS fixo de 1)
entradas = np.array([[1,-1,-1],
                     [1, 1,-1],
                     [1,-1, 1],
                     [1, 1, 1]])

# Saída dos dados. Resulta 1 se uma das duas entradas for 1          
saidas = np.array([[-1,
                     1,
                     1,
                     1]]).T


# Segundo exemplo. XOR, não linearmente separável
# Dados de entrada representando o operador logico XOR (com o BIAS fixo de 1)
#entradas = np.array([[1,-1,-1],
#                     [1, 1,-1],
#                     [1,-1, 1],
#                     [1, 1, 1]])
#
## Saída dos dados. Resulta 1 se as entradas forem diferentes.          
#saidas = np.array([[-1,
#                     1,
#                     1,
#                    -1]]).T

# Inicializa os pesos aleatoriamente com média 0
pesos = 2 * np.random.random((3,1)) - 1
print ("\nPesos aleatórios antes do treino: \n", pesos)

# Loop de treino com limite de 100 ajustes
for i in range(100):

    for entrada,saidaDesejada in zip(entradas, saidas):
        
		 # Alimenta (feedforward) e calcula o somatório da Adaline
        somatorio = (entrada[0]*pesos[0]) + (entrada[1]*pesos[1]) + (entrada[2]*pesos[2])

        # Processa a saída atraves da função degrau
        saidaAdaline = step(somatorio)

        # Calcula o erro gerado
        erro = saidaDesejada - saidaAdaline
        
        # Armazena os erros e os pesos
        erros.append(erro)
        bias.append (pesos[0][0])
        peso1.append(pesos[1][0])
        peso2.append(pesos[2][0])
        
		# Atualiza os pesos de acordo com a regra do Delta
        pesos[0] = pesos[0] + taxaAprendizado * erro * entrada[0]
        pesos[1] = pesos[1] + taxaAprendizado * erro * entrada[1]
        pesos[2] = pesos[2] + taxaAprendizado * erro * entrada[2]

print ("\nNovos pesos após o treino: \n", pesos, "\n")

for entrada,saidaDesejada in zip(entradas, saidas):
    
    # Alimenta a entrada para frente (feedforward) e calcula a saída da Adaline
    somatorio = (entrada[0]*pesos[0]) + (entrada[1]*pesos[1]) + (entrada[2]*pesos[2])

    # Processa a saída atraves da função degrau
    saidaAdaline = step(somatorio)

    print ("Saída calculada: ", saidaAdaline, "  Saída desejada: ", saidaDesejada)

# Plota os erros durante o treinamento
ax = plt.subplot(111)
ax.set_xscale("log")
#ax.set_ylim([-2,2])
plt.plot(erros,'#000000')
plt.legend(('Erro',),shadow=True)
plt.title("Erros durante treino da Adaline")
plt.xlabel('Iteração')
plt.ylabel('Valor')
plt.show()

# Plota as variações dos pesos durante o treino
ax = plt.subplot(111)
ax.set_xscale("log")
#ax.plot(erros, c='#000000', label='Erro', alpha=0.3)
plt.plot(bias,'r',peso1,'g',peso2,'b')
plt.legend(('Bias','Peso 1', 'Peso 2'),shadow=True)
plt.title("Ajuste dos pesos durante treino da Adaline")
plt.xlabel('Iteração')
plt.ylabel('Valor')
plt.show()