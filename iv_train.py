import argparse
import os
import sys
import time

import tensorflow as tf


import mwlib as mb
import numpy as np
import matplotlib.pyplot as plt

# Basic model parameters as external flags.
FLAGS = None


def evaluation(input_placeholder, output_placeholder, prediction, data_set):
    """Evaluate the quality of the output.

    Args:
      input_placeholder: placeholder for input tensor
      output_placeholder: placeholder for output tensor
      prediction: ground truth value
      data_set: dataset on which evaluation will be performed

    Returns:
      test_loss: a tensor for test loss
      feed_dict: feed_dict for the tensor to be evaluated
    """

    #the first feed element is the design, which is the output of the inverse net
    feed_dict = mb.fill_feed_dict(data_set, output_placeholder, input_placeholder)


    # Add to the Graph the Ops for loss calculation.
    test_loss = mb.loss(prediction, output_placeholder)

    return test_loss, feed_dict





def run_training():
    """Train network for a number of steps."""

    #trainning data after normalization
    training_data_LL = np.genfromtxt(FLAGS.working_dir+'data/train_data_norm_LL.csv', delimiter=',', skip_header=1)[:,1:]
    training_data_LR = np.genfromtxt(FLAGS.working_dir+'data/train_data_norm_LR.csv', delimiter=',', skip_header=1)[:,1:]
    training_data_RL = np.genfromtxt(FLAGS.working_dir+'data/train_data_norm_RL.csv', delimiter=',', skip_header=1)[:,1:]
    #test data after normalization
    test_data_LL = np.genfromtxt(FLAGS.working_dir+'data/test_data_norm_LL.csv', delimiter=',', skip_header=1)[:,1:]
    test_data_LR = np.genfromtxt(FLAGS.working_dir+'data/test_data_norm_LR.csv', delimiter=',', skip_header=1)[:,1:]
    test_data_RL = np.genfromtxt(FLAGS.working_dir+'data/test_data_norm_RL.csv', delimiter=',', skip_header=1)[:,1:]

    #stack training and test data
    training_data = np.stack((training_data_LR, training_data_LL, training_data_RL), axis=-1)
    test_data = np.stack((test_data_LR, test_data_LL, test_data_RL), axis=-1)

    #number of epochs to save graph
    save_period = 20



    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
    # Generate placeholders for the designs and spectra.
        design_placeholder, spec_placeholder = mb.placeholder_inputs() #should be FLAG.batch_size during training process

        #Build a Graph of inverse design neural network
        iv_prediction = mb.iv_inference(spec_placeholder,
                             FLAGS.iv_fc1,
                             FLAGS.iv_fc2,
                             FLAGS.iv_conv_channel)


        # Add to the Graph the Ops for loss calculation.
        iv_loss = mb.loss(iv_prediction, design_placeholder)

        # Add to the Graph the Ops that calculate and apply gradients.
        train_op = mb.training(iv_loss, FLAGS.learning_rate, FLAGS.momentum)

        # Add the variable initializer Op.
        init = tf.global_variables_initializer()

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()

        # Create a session for running Ops on the Graph.
        sess = tf.Session()

        #create tensors to be incorporated in summary

        iv_train_loss, iv_train_feed_dict = evaluation(input_placeholder=spec_placeholder, output_placeholder=design_placeholder,
                                             prediction=iv_prediction, data_set=training_data)
        iv_test_loss, iv_test_feed_dict = evaluation(input_placeholder=spec_placeholder, output_placeholder=design_placeholder,
                                           prediction=iv_prediction, data_set=test_data)

        # Add a scalar summary for the snapshot train loss.
        summary_iv_train_loss = tf.summary.scalar('iv_train_loss', iv_train_loss)
        # Add a scalar summary for the snapshot test loss.
        summary_iv_test_loss = tf.summary.scalar('iv_test_loss', iv_test_loss)

        # Add a scalar summary for the snapshot batch train loss (total loss).
        summary_batch_train_loss = tf.summary.scalar('batch_train_loss', iv_loss)

        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer = tf.summary.FileWriter(FLAGS.working_dir, sess.graph)

        # And then after everything is built:
        # Run the Op to initialize the variables.
        sess.run(init)

        # Start the training loop.
        for step in range(FLAGS.epoch):
            start_time = time.time()

            np.random.shuffle(training_data)  # shuffle training data after on epoch

            for batch_idx in range(int(training_data.shape[0]/FLAGS.batch_size)):

                #generate data to be fed according  to batch size and total training sample number
                fed_data = training_data[batch_idx*FLAGS.batch_size:(batch_idx+1)*FLAGS.batch_size,:]

                # Fill a feed dictionary with the actual set of designs and spectra for this particular training step.

                feed_dict_train = mb.fill_feed_dict(fed_data,
                                 design_placeholder,
                                 spec_placeholder)

                # Run one step of the model.  The return values are the activations
                # from the `train_op` (which is discarded) and the `loss` Op.
                _, loss_value = sess.run([train_op, iv_loss], feed_dict=feed_dict_train)

            duration = time.time() - start_time

            # print current loss value on training set
            if step % 10 == 0:
                # Write the summaries for batch train loss
                summary_batch_train_loss_str = sess.run(summary_batch_train_loss, feed_dict=feed_dict_train)
                summary_writer.add_summary(summary_batch_train_loss_str, global_step=step)

                # Flushes the event file to disk.
                summary_writer.flush()

                print('epoch %d: training loss = %.4f (%.3f sec)' % (step, loss_value, duration))

            #record training loss and test loss periodically
            if step % save_period == 0:

                # Write the summaries.
                summary_iv_train_loss_str = sess.run(summary_iv_train_loss, feed_dict=iv_train_feed_dict)
                summary_iv_test_loss_str = sess.run(summary_iv_test_loss, feed_dict=iv_test_feed_dict)

                summary_writer.add_summary(summary_iv_train_loss_str, global_step=step)
                summary_writer.add_summary(summary_iv_test_loss_str, global_step=step)

                # Flushes the event file to disk.
                summary_writer.flush()

            if (step+1) % 500 == 0 or step == FLAGS.epoch-1:
                # save model periodically
                checkpoint_file = os.path.join(FLAGS.working_dir, 'iv_model.ckpt')
                saver.save(sess, checkpoint_file, global_step=step+1)

    return
###############################################################################


def main(_):

    start = time.clock()

    run_training()

    end = time.clock()
    print('Running time: %s Seconds' % (end - start))


# main program starts here!!!
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.1,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--momentum',
        type=float,
        default=0.9,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--epoch',
        type=int,
        default=2000,   # set epoch numbers, 2000 
        help='Number of steps to run trainer.'
     )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1000,
        help='Number of steps to run trainer.'
     )
    parser.add_argument(
        '--iv_conv_channel',
        type=int,
        default=mb.d_iv_conv_channel,
        help='Number of channels in convolutional layers of the inverse net.'
    )
    parser.add_argument(
        '--iv_fc1',
        type=int,
        default=mb.d_iv_fc1,
        help='Number of units in first fully connected layer in inverse net.'
    )
    parser.add_argument(
        '--iv_fc2',
        type=int,
        default=mb.d_iv_fc2,
        help='Number of units in second fully connected layer in inverse net.'
    )
    parser.add_argument(
        '--working_dir',
        type=str,
        default=mb.d_wdir,
        help='Directory to put the log data.'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)