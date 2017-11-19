import numpy as np
import tensorflow as tf

class TextCNN(object):
    """
    Rede Neural Convolucional para classificacao de texto
    """
    def __init__(self, sequence_length, num_classes, vocab_size,
                    embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
        """
        Objeto da rede

        :param sequence-length: tamanho da sentenca
        :param num_classes: quantidade de outputs
        :param vocab_size: tamanho do vocabulario, qtde de palavras diferentes nos exemplos
        :param embedding_size: tamanho dos embeddings
        :param filter_sizes: array de qtde de palavras que os filtros vao passar
        :param num_filters: qtde de filtros usados
        """
        #placeholders de input, output e dropout da rede
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        l2_loss = tf.constant(0.0)

        #embedding, lista das palavras no vocabulario
        with tf.device("/cpu:0"), tf.name_scope("embedding"):
            W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="W")
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x) #cria lista das palavras, lookup table
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1) #cria outra dimensao para usar no conv2d

        #criando as camadas de filtro
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conf-maxpool-%s" % filter_size):
                #camada de convolucao
                #cada filtro tem um tamanho diferente e passa por todas as palavras
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters], name="b"))
                #passa o filtro sem fazer padding, diminui o tamanho da saida
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                #aplica relu
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                #max pool no resultado da convolucao + relu
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="pool")
                #junta todos os resultados de pool
                pooled_outputs.append(pooled)
    
        #concatena todos os pools em um vetor so
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        #camada do dropout das saidas
        with tf.name_scope("dropout"):
            self.h_dropout = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        #camada de saida
        with tf.name_scope("output"):
            W = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            #computa norma l2 para evitar overfitting
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            #faz o classico xW + b com matmul
            self.scores = tf.nn.xw_plus_b(self.h_dropout, W, b, name="scores")
            #pega o valor maior das predicoes de saida
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        #calculando perda
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        #acuracea
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")