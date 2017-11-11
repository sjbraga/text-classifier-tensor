#Tensorflow Hello World file
#%%
from __future__ import print_function

import os

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#criando tensores de constante
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0)
#print(node1, node2)

#criando soma
node3 = tf.add(node1, node2)

#cria a sessao que avalia os tensores
sess = tf.Session()

#para avaliar os tensores criados
print('run node 3', sess.run(node3))

#%%
#placeholder: variaveis
#cria dois parametros de entrada e define a operação
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
#shortcut da soma tf.add
adder_node = a + b

#roda a sessao passando os valores para as variaiveis
print(sess.run(adder_node, {a: 4, b:7.5}))
#passando valores como vetores
print(sess.run(adder_node, {a:[1,3], b:[2,4]}))

#%%
#coloca mais uma operação depois da soma
add_and_triple = adder_node * 3
print(sess.run(add_and_triple, {a: 4, b: 9}))

#%%
#criando variaveis com valores iniciais
#variaveis nao sao inicializadas automaticamente
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)

#cria um modelo linear
linear_model = W*x + b

#inicializa variaveis
init = tf.global_variables_initializer()
sess.run(init)

#executa linear model para varios valores de x
print(sess.run(linear_model, {x: [1,2,3,5]}))

#%%
#para avaliar o desempenho do modelo, precisa criar uma função de perda
#criando placeholder de y = saidas desejadas
y = tf.placeholder(tf.float32)

#funçao de perda root mean squared
#linear_model - y cria vetor com as diferenças da saida
# com o valor esperado e eleva ao quadrado
squared_deltas = tf.square(linear_model - y)
#função de perda
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

#%%
#coloca novos valores em variaveis
fixW = tf.assign(W, [-1.])
fixB = tf.assign(b, [1.])
sess.run([fixW, fixB])
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))
