# -*- coding: utf-8 -*-

import tensorflow as tf
import argparse
import utils


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--checkpoint_dir', type=str, default='save',
                        help='model directory to store checkpointed models')
    parser.add_argument('--text', type=str, default=u' ',
                        help='prime text')
    args = parser.parse_args()
    sample(args)


def sample(args):
    print 'Loading data'
    x, y, vocabulary, vocabulary_inv = utils.load_data()

    text = [list(args.text)]
    sentences_padded = utils.pad_sentences(text, maxlen=x.shape[1])
    raw_x, dummy_y = utils.build_input_data(sentences_padded, [0], vocabulary)

    checkpoint_file = tf.train.latest_checkpoint(args.checkpoint_dir)
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

            predicted_result = sess.run(predictions, {input_x: raw_x, dropout_keep_prob: 1.0})
            if (predicted_result[0] == 0):
                print args.text + ": negative"
            else:
                print args.text + ": positive"


if __name__ == '__main__':
    main()
