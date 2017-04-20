# -*- coding: utf-8 -*-

import tensorflow as tf
import utils
from flask import Flask, request, jsonify

# Server Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

print 'Loading data'
x, y, vocabulary, vocabulary_inv = utils.load_data()

"""
Restore the model
"""
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    sess = tf.Session()
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    if not request.json or not 'text' in request.json:
        abort(400)

    text = request.json['text']
    raw_x = utils.sentence_to_index(text, vocabulary, x.shape[1])
    predicted_results = sess.run(predictions, {input_x: raw_x, dropout_keep_prob: 1.0})

    return jsonify({'result': predicted_results[0]})


if __name__ == '__main__':
    app.run(debug=True)
