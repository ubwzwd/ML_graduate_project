import argparse
import os
import sys

import tensorflow as tf
import mwlib as mb
import numpy as np
import matplotlib.pyplot as plt

def main(_):
    tf.reset_default_graph()

    design_placeholder, spec_placeholder = mb.placeholder_inputs() #should be FLAG.batch_size during training process

    # Build a Graph of bi-direction neural network
    iv_prediction = mb.iv_inference(spec_placeholder,
                                    FLAGS.iv_fc1,
                                    FLAGS.iv_fc2,
                                    FLAGS.iv_conv_channel)

    # Create a session for running Ops on the Graph.
    sess = tf.Session()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
    path = os.path.join(FLAGS.working_dir + 'iv_model', 'iv_model.ckpt-2000') 
    saver.restore(sess, path)

    run_plot_evaluate(iv_prediction, design_placeholder, spec_placeholder, session=sess, test_range=range(100,110))



def get_prediction(test_input, model, design_placeholder, spec_placeholder, session):
    with tf.Graph().as_default():

        feed_dict = mb.fill_feed_dict(test_input,
                                   design_placeholder,
                                   spec_placeholder)

        predicted_design = session.run(model, feed_dict=feed_dict)
    return predicted_design

def run_plot_evaluate(model, design_placeholder, spec_placeholder, session, test_range=range(3)):
    # plot data is the test data before normalizaition, only need one slice of the 3 polarization for design parameters
    plot_data = np.genfromtxt(FLAGS.working_dir + 'data/test_data_LR.csv', delimiter=',', skip_header=1)[:, 1:]

    # test data is the test data after normalizaition, only need normalized design parameters
    test_data_LR = np.genfromtxt(FLAGS.working_dir + 'data/test_data_norm_LR.csv', delimiter=',', skip_header=1)[:, 1:]
    test_data_LL = np.genfromtxt(FLAGS.working_dir + 'data/test_data_norm_LL.csv', delimiter=',', skip_header=1)[:, 1:]
    test_data_RL = np.genfromtxt(FLAGS.working_dir + 'data/test_data_norm_RL.csv', delimiter=',', skip_header=1)[:, 1:]
    test_data = np.stack((test_data_LR, test_data_LL, test_data_RL), axis=-1)

    plot_input = plot_data[test_range, :]
    test_input = test_data[test_range, :, :]

    prediction = get_prediction(test_input, model, design_placeholder, spec_placeholder, session)

    prediction_recover = np.zeros([np.shape(prediction)[0] ,5])
    prediction_recover[:, 0] = mb.mwnorm(prediction[:, 0], label='top_length', reverse=True)
    prediction_recover[:, 1] = mb.mwnorm(prediction[:, 1], label='bottom_length', reverse=True)
    prediction_recover[:, 2] = mb.mwnorm(prediction[:, 2], label='top_spacer', reverse=True)
    prediction_recover[:, 3] = mb.mwnorm(prediction[:, 3], label='bottom_spacer', reverse=True)
    prediction_recover[:, 4] = mb.mwnorm(prediction[:, 3], label='angle', reverse=True)

    for i, _ in enumerate(test_range):
        mb.iv_plot(plot_input[i,:], prediction_recover[i,:])
    plt.show()


# Basic model parameters as external flags, should be the same as in forward_net.py
FLAGS = None

# main program starts here!!!
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--iv_conv_channel',
        type=int,
        default = mb.d_iv_conv_channel,
        help='Number of channels in convolutional layers of the inverse net.'
    )
    parser.add_argument(
        '--iv_fc1',
        type=int,
        default = mb.d_iv_fc1,
        help='Number of units in first fully connected layer in forward net.'
    )
    parser.add_argument(
        '--iv_fc2',
        type=int,
        default = mb.d_iv_fc2,
        help='Number of units in second fully connected layer in forward net.'
    )
    parser.add_argument(
        '--working_dir',
        type=str,
        default = mb.d_wdir,
        help='Directory to put the log data.'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)