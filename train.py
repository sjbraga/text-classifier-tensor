import datetime
import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn

import data_helper as dh
from cnn_text import TextCNN

# Parametros -----------------------------------------------------

# Parametros de dados
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Porcentagem dos dados usados para validacao")
tf.flags.DEFINE_string("positive_data_file", "./dataset/rt-polarity.pos", "Caminho do arquivo de reviews positivas")
tf.flags.DEFINE_string("negative_data_file", "./dataset/rt-polarity.neg", "Caminho do arquivo de reviews negativas")

# Parametros do modelo
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionalidade de character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Tamanho dos filtros separados por virgula (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Quantidade de filtros por tipo (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Probabilidade do dropout (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Parametros de treinamento
tf.flags.DEFINE_integer("batch_size", 64, "Tamanho do batch de treinamento (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Numero de epocas de treinamento (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Validar o modelo apos quantas iteracoes (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Salvar modelo apos quantas iteracoes (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Numero de checkpoints mantidos (default: 5)")

# Parametros do tensorflow
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", True, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParametros:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Carregando e processando dados -------------------------------------------

print("Carregando dados...")
x_text, y = dh.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)


# cria vocabulario
# maior numero de palavras num unico documento
max_document_length = max([len(x.split(" ")) for x in x_text])
# gerando lista de palavras
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
# cria arrays de palavras para input
x = np.array(list(vocab_processor.fit_transform(x_text)))

# seleciona parte dos dados randomicamente
np.random.seed(10)
# seleciona indices
shuffle_indices = np.random.permutation(np.arange(len(y)))
# pega x e y com os indices selecionados
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# split do dataset em treino e teste
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
print("Tamanho do vocabulario: {:d}",format(len(vocab_processor.vocabulary_)))
print("Train/dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))


# Treino -----------------------------------------------------------------

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                    log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)

    with sess.as_default():
        cnn = TextCNN(
            sequence_length=x_train.shape[1],
            num_classes=y_train.shape[1],
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # definindo treinamento
        # contagem das iteracoes de treinamento
        global_step = tf.Variable(0, name="global_step", trainable=False)
        # usando adam ao inves de gradient descent
        optimizer = tf.train.AdamOptimizer(1e-3)
        # calcula os gradientes (derivadas)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        # atualiza os pesos de acordo com o gradiente
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # gerando sumarios dos gradientes
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # gerando sumario do treino e modelo
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Escrevendo em {}\n".format(out_dir))

        #sumario de perda e acuracea
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        #sumario de treino
        train_summary_op = tf.summary.merge([loss_summary, acc_summary])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph_def)

        #sumario dev
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph_def)

        #checkpoint -> salvando parametros do modelo
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        #salva o vocabulario
        vocab_processor.save(os.path.join(out_dir, "vocab"))

        #inicializa variaveis
        sess.run(tf.global_variables_initializer())

        def training_step(x_batch, y_batch):
            """
            Execucao de uma iteracao de treinamento
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }

            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step: {}, loss: {:g}, acc: {:g},".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Avalia o modelo no momento do treinamento
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: 1.0 #desliga dropout para avaliar
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step: {}, loss: {:g}, acc: {:g},".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

        #gerando os batches
        batches = dh.batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
        #iteracoes de treino
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            training_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\n Avaliacao:")
                dev_step(x_dev, y_dev, writer=dev_summary_writer)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Checkpoint do modelo salvo em {}".format(path))

