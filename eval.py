import csv
import datetime
import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn

import data_helper
from cnn_text import TextCNN

# Parametros
# ==================================================

# Parametros de dados
tf.flags.DEFINE_string("positive_data_file", "./dataset/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./dataset/rt-polarity.neg", "Data source for the negative data.")

# Parametros de avaliacao
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", os.path.join(os.path.curdir, "runs/1510531385/checkpoints"), "Checkpoint directory from training run")
tf.flags.DEFINE_string("vocab_dir", "runs/1510531385", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Parametros do tensorflow
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParametros:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

if FLAGS.eval_train:
    x_raw, y_test = data_helper.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
    y_test = np.argmax(y_test, axis=1)
else:
    x_raw = ["a masterpiece four years in the making", "everything is off."]
    y_test = [1, 0]

# carregando vocabulario do treino
vocab_path = os.path.join(FLAGS.vocab_dir, "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
# transformando frases fixas em formato de array de palavras
x_test = np.array(list(vocab_processor.transform(x_raw)))

print("\nAvaliando...\n")

# Avaliacao
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()

with graph.as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # carrega o grafo da rede e restaura variaveis
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # pega placeholder pelo nome (comando with ao criar o grafo)
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # pega o tensor que vai avaliar -> a saida
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # gera o batch com as duas frases de avaliacao para uma epoca
        batches = data_helper.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, feed_dict={
                input_x: x_test_batch, dropout_keep_prob: 1.0}) #desliga o dropout para testar
            all_predictions = np.concatenate([all_predictions, batch_predictions])

if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test))
    print("Exemplos de teste: {}".format(len(y_test)))
    print("Acuracia: {:g}".format(correct_predictions/float(len(y_test))))


predictions_readable = np.column_stack((np.array(x_raw), all_predictions))
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
print("Salvando avaliacao em {0}".format(out_path))

with open(out_path, 'w') as f:
    csv.writer(f).writerows(predictions_readable)
