import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

frequency_range = np.linspace(30,80,201)

#define linear normalization function
def mwnorm(inputdata, label, reverse=False): #normalize to [-1,1], label='dspacer', 'width', 'index', 'Ef', reverse=True, do the opposite
    if label == 'top_length':
        label_max = 1600.
        label_min = 1000.
    if label == 'bottom_length':
        label_max = 1600
        label_min = 1000
    if label == 'top_spacer':
        label_max = 600.
        label_min = 200.
    if label == 'bottom_spacer':
        label_max = 600
        label_min = 200
    if label == 'angle':
        label_max = 180
        label_min = 0
    if reverse == False: return 2.0 / (label_max - label_min) * (inputdata - label_min) - 1
    if reverse == True: return (inputdata + 1)*(label_max - label_min) / 2.0 + label_min

def fw_plot(test_input, predicted_spectra):
    # plot results, input is a 1*NUM_SPEC*3 vector!
    label = ', t_l=' + str(test_input[0, 0]) + ', b_l=' + str(test_input[1, 0]) + '\n t_s=' + str(
        test_input[2, 0]) + ', b_s=' + str(test_input[3, 0]) + ', arg=' + str(test_input[4, 0])

    prediction_LR = predicted_spectra[:, 0]
    prediction_LL = predicted_spectra[:, 1]
    prediction_RL = predicted_spectra[:, 2]

    true_LR = test_input[5:, 0]
    true_LL = test_input[5:, 1]
    true_RL = test_input[5:, 2]



    # draw expected spectrum
    plt.figure()
    plt.plot(frequency_range, true_LR, 'r--', lw=2, label='expected LR' + label)
    plt.plot(frequency_range, true_LL, 'b--', lw=2, label='expected LL' + label)
    plt.plot(frequency_range, true_RL, 'g--', lw=2, label='expected RL' + label)

    # draw predicted spectrum
    plt.plot(frequency_range, prediction_LR, 'r-', lw=2, label='predicted LR' + label)
    plt.plot(frequency_range, prediction_LL, 'b-', lw=2, label='predicted LL' + label)
    plt.plot(frequency_range, prediction_RL, 'g-', lw=2, label='predicted RL' + label)

    plt.xlabel('frequency (THz)')
    plt.ylabel('reflectance')
    plt.legend()

    return

def iv_plot(test_input, predicted_design):
    # plot results, input is a 1*n vector!
    plt.figure()
    x_value = np.arange(5) + 1

    my_xticks = ['top_length', 'bottom_length', 'top_spacer', 'bottom_spacer', 'angle']
    plt.xticks(x_value, my_xticks)

    exp = test_input[:5]
    pre = predicted_design
    exp_plot = plt.bar(x_value - 0.2, exp, width=0.4, facecolor='lightskyblue', edgecolor='lightskyblue')
    # width: width of the post
    pre_plot = plt.bar(x_value + 0.2, pre, width=0.4, facecolor='yellowgreen', edgecolor='yellowgreen')

    plt.legend((exp_plot[0], pre_plot[0]), ('expected', 'predicted'))

    for x, y in zip(x_value, exp):
        plt.text(x - 0.2, y + 0.05, '%.2f' % y, ha='center', va='bottom')

    for x, y in zip(x_value, pre):
        plt.text(x + 0.2, y + 0.05, '%.2f' % y, ha='center', va='bottom')

    # plt.ylim(0, 1.25)

    # plt.show()

    return


###########################################################################
# The output spectrum data is discretized into 51 data points
SPEC_NUM = 201
# The design parameters is a 1*4 vector
DESIGN_NUM = 5
#default model parameters
d_fw_tensor = 50
d_fw_fc1 = 67*3
d_fw_conv_channel = 20
d_iv_conv_channel = 10
d_iv_fc1 = 100
d_iv_fc2 = 100
d_wdir = 'E:/python/ML_project_2.0/temp_data/'

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=1.0 / math.sqrt(float(shape[0])), name='weights', dtype='float')
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0, shape=shape, name='biases', dtype='float')
  return tf.Variable(initial)

def conv2d(x, w):
  """conv2d returns a 2d convolution layer with full stride.
     For the most common case of the same horizontal and vertices strides, strides = [1, stride, stride, 1].
  """
  return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def conv2d_transpose(x, w, output_shape):
  """conv2d_transposed is the transpose convolution applied to up sampling input
     For the most common case of the same horizontal and vertices strides, strides = [1, stride, stride, 1].
     tf.nn.conv2d_transpose(value, filter, output_shape, strides, padding='SAME', data_format='NHWC', name=None)
  """

  return tf.nn.conv2d_transpose(x, w, output_shape, strides=[1, 3, 1, 1], padding='SAME')


def max_pool_3x1(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='SAME')


def fw_inference(design_par, fw_tensor, fw_fc1, fw_conv_channel):
    """Build the forward model for inference.

    Args:
      design_par: design parameter placeholder, from inputs().
      fw_tensor: Size of the tensor layer.
      fw_fc1: Size of the first fully connected layer

    Returns:
      linear: Output tensor for as spectrum
    """
    # tensor layer
    with tf.name_scope('fw_tensor'):
        slice_list = list()
        
        tensors = tf.Variable(
            tf.truncated_normal([DESIGN_NUM, DESIGN_NUM, fw_tensor],
                                stddev=1.0 / math.sqrt(float(DESIGN_NUM)**2)))
        for slice in range(fw_tensor):
            slice_list.append(tf.reduce_sum(tf.matmul(design_par, tensors[:, :, slice]) * design_par, axis=1))

        # tensor product term with input data
        tensor_term = tf.stack(slice_list, axis=1)

        weights = weight_variable([DESIGN_NUM, fw_tensor])
        biases = bias_variable([fw_tensor])
        fw_tensor_out = tf.nn.relu(tensor_term + tf.matmul(design_par, weights) + biases)
  # first fully connected layer
    with tf.name_scope('fw_fc1'):
        weights = weight_variable([fw_tensor, fw_fc1])
        biases = bias_variable([fw_fc1])
        fw_fc1_out = tf.nn.relu(tf.matmul(fw_tensor_out, weights) + biases)
        fw_fc1_out = tf.reshape(fw_fc1_out, [-1, int(SPEC_NUM/3), 3, 1]) #output one slice is 67*3 for up sampling
  # transposed convolutional layer
    with tf.name_scope('fw_conv1_tranpose'):
        # attention! weight format is [height, width, output_channels, in_channels] for transpose convolution
        weights_conv = weight_variable([3, 3, fw_conv_channel, 1])
        biases_conv = bias_variable([fw_conv_channel])
        out_shape = [tf.shape(fw_fc1_out)[0], SPEC_NUM, 3, fw_conv_channel] # output shape of the transposed convolution layer
        fw_conv1_out = tf.nn.relu(conv2d_transpose(fw_fc1_out, weights_conv, out_shape) + biases_conv)
  # Linear out put, 1*1 convolutional output layer
    with tf.name_scope('fw_out'):
        weights_conv = weight_variable([3, 3, fw_conv_channel, 1])
        biases_conv = bias_variable([1])
        fw_out = conv2d(fw_conv1_out, weights_conv) + biases_conv
    return tf.squeeze(fw_out, [3]) #squeeze to 3 dimensional data


def iv_inference(spectrum, iv_fc1, iv_fc2, iv_conv_channel):
    """Build the inverse model for inference.

    """
    #The first two dimensions are the patch size, the next is the number of input channels,
    # and the last is the number of output channels. We will also have a bias vector with a component for each output channel.
    with tf.name_scope('iv_conv1'):
        weights_conv = weight_variable([3, 3, 1, iv_conv_channel])
        biases_conv = bias_variable([iv_conv_channel])
        spec_fed = tf.reshape(spectrum, [-1, SPEC_NUM, 3, 1])
        iv_conv1_out = tf.nn.relu(conv2d(spec_fed, weights_conv) + biases_conv)

    # Pooling layer - downsamples by 2X.
    with tf.name_scope('iv_pool1'):
        iv_pool1_out = max_pool_3x1(iv_conv1_out)

  # first fully connected layer, input is 17*1*16
    with tf.name_scope('iv_fc1'):
        weights = weight_variable([int(SPEC_NUM/3) * 1 * iv_conv_channel, iv_fc1])
        biases = bias_variable([iv_fc1])
        iv_pool_flat = tf.reshape(iv_pool1_out, [-1, int(SPEC_NUM/3) * 1 * iv_conv_channel])
        iv_fc1_out = tf.nn.relu(tf.matmul(iv_pool_flat, weights) + biases)
  # second fully connected layer
    with tf.name_scope('iv_fc2'):
        weights = weight_variable([iv_fc1, iv_fc2])
        biases = bias_variable([iv_fc2])
        iv_fc2_out = tf.nn.relu(tf.matmul(iv_fc1_out, weights) + biases)
  # Linear
    with tf.name_scope('iv_out'):
        weights = weight_variable([iv_fc2, DESIGN_NUM])
        biases = bias_variable([DESIGN_NUM])
        iv_out = tf.matmul(iv_fc2_out, weights) + biases
    return iv_out


def loss(pred_value, true_value):
    """Calculates the loss from the output and the true value.

    Args:
      pred_value: output tensor
      true_value: true_value tensor

    Returns:
      loss: Loss tensor of type float.
    """

    # error = tf.losses.mean_squared_error(labels=true_value, predictions=out_spec, weights=1)

    error = tf.reduce_mean(tf.squared_difference(true_value, pred_value))

    return error


def training(loss, learning_rate, momentum):
    """Sets up the training Ops.

    Creates an optimizer and applies the gradients to all trainable variables.

    The Op returned by this function is what must be passed to the
  `  sess.run()` call to cause the model to train.

    Args:
      loss: Loss tensor, from loss().
      learning_rate: The learning rate to use for gradient descent.
      momentun: The momentum used in SGD

    Returns:
      train_op: The Op for training.
    """

    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


##################################################
def placeholder_inputs(batch_size = None):
    """Generate placeholder variables to represent the input tensors.

    These placeholders are used as inputs by the rest of the model building
    code and will be fed from the loaded data in the .run() loop, below.

    Args:
    batch_size: The batch size will be baked into both placeholders.

    Returns:
      design_placeholder: Design input placeholder, 5 design parameters in total
      spec_placeholder: True spec value placeholder, containing 3 spectra, each has 201 data points
    """
    # Note that the shapes of the placeholders match the shapes of the input design parameters and
    # output spectrum tensors, except the first dimension is now batch_size
    # rather than the full size of the train or test data sets.
    design_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                         DESIGN_NUM))
    spec_placeholder = tf.placeholder(tf.float32, shape=(batch_size, SPEC_NUM, 3))
    return design_placeholder, spec_placeholder


def fill_feed_dict(batch_data, input_pl, output_pl):
    """Fills the feed_dict for training the given step.

    A feed_dict takes the form of:
    feed_dict = {
        <placeholder>: <tensor of values to be passed for placeholder>,
      ....
    }

    Args:
      batch_data: np array containing a batch of data, first dimension is batch size, second dimension is design parameter/spectrum
      data for LR, LL, RR respectively, third dimension is the 3 polarization condition. second dimension is the same for all slices
      in the third dimension
      input_pl: The input placeholder, from placeholder_inputs(), for forward net, it's the 5 design parameters
      output_pl: The output placeholder, from placeholder_inputs(), for forward net, it's the 3 spectra of 201 data points each.

    Returns:
      feed_dict: The feed dictionary mapping from placeholders to values.
    """
    # Create the feed_dict for the placeholders filled with data in the batch


    input_feed = batch_data[:, :5, 0]  #take the design parameter of one slice of LR, LL, RL is enough
    output_feed = batch_data[:, 5:, :]


    feed_dict = {
        input_pl: input_feed,
        output_pl: output_feed,
    }
    return feed_dict

