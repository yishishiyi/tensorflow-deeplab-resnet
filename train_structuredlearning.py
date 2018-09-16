"""Training script for the DeepLab-ResNet network on the PASCAL VOC dataset
   for semantic image segmentation.

This script trains the model using augmented PASCAL VOC,
which contains approximately 10000 images for training and 1500 images for validation.
"""

from __future__ import print_function

import argparse
from datetime import datetime
import os
import sys
import time

import tensorflow as tf
import numpy as np

from deeplab_resnet import DeepLabResNetStructuredLearningModel, ImageReader, decode_labels, inv_preprocess, prepare_label

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

BATCH_SIZE = 1
DATA_DIRECTORY = './data/VOC2012_Aug/VOCdevkit/VOC2012'
DATA_LIST_PATH = './dataset/small_batch.txt'
VAL_DATA_LIST_PATH = './dataset/val_small_batch.txt'
IGNORE_LABEL = 255
INPUT_SIZE = '321,321'
LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
NUM_CLASSES = 21
NUM_STEPS = 20001
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = './ckpt/deeplab_resnet_tf/deeplab_resnet_init.ckpt'
SAVE_NUM_IMAGES = 1
SAVE_PRED_EVERY = 10
SUMMARY_EVERY = 2
SNAPSHOT_DIR = './snapshots/'
WEIGHT_DECAY = 0.0005
GPU_ID = '0'
EMBEDDING_SIZE = 512
ASPP = True
CRN = True


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--val-data-list", type=str, default=VAL_DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save checkpoint every often.")
    parser.add_argument("--summary-every", type=int, default=SUMMARY_EVERY,
                        help="Save summaries every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu-id", type=str, default=GPU_ID,
                        help="GPU id, -1 means cpu.")
    parser.add_argument("--embedding-size", type=int, default=EMBEDDING_SIZE,
                        help="embedding-size before refinement.")
    parser.add_argument("--ASPP", type=bool, default=ASPP,
                        help="Whether use Atrous Spatial Pyramid Pooling")
    parser.add_argument("--CRN", type=bool, default=CRN,
                        help="Wether use CRN to refine output")

    return parser.parse_args()


def save(saver, sess, logdir, step):
    '''Save weights.

    Args:
      saver: TensorFlow Saver object.
      sess: TensorFlow session.
      logdir: path to the snapshots directory.
      step: current training step.
    '''
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    saver.save(sess, checkpoint_path, global_step=step)
    print('The checkpoint has been created.')


def load(saver, sess, ckpt_path):
    '''Load trained weights.

    Args:
      saver: TensorFlow Saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    '''
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))


def main():
    """Create the model and start the training."""
    args = get_arguments()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)

    tf.set_random_seed(args.random_seed)

    # Create queue coordinator.
    coord = tf.train.Coordinator()

    # Load reader.
    with tf.name_scope("create_inputs"):
        reader = ImageReader(
            args.data_dir,
            args.data_list,
            input_size,
            args.random_scale,
            args.random_mirror,
            args.ignore_label,
            IMG_MEAN,
            coord)
        val_reader = ImageReader(
            args.data_dir,
            args.val_data_list,
            None,
            False,
            False,
            args.ignore_label,
            IMG_MEAN,
            coord)
        image_batch, label_batch = reader.dequeue(args.batch_size)
        print("image_batch", image_batch.get_shape().as_list())
        print("label_batch", label_batch.get_shape().as_list())
        val_image_batch, val_label_batch = tf.expand_dims(val_reader.image, dim=0), tf.expand_dims(val_reader.label, dim=0)

    # Create network.
    # For a small batch size, it is better to keep
    # the statistics of the BN layers (running means and variances)
    # frozen, and to not update the values provided by the pre-trained model.
    # If is_training=True, the statistics will be updated during the training.
    # Note that is_training=False still updates BN parameters gamma (scale) and beta (offset)
    # if they are presented in var_list of the optimiser definition.
    net = DeepLabResNetStructuredLearningModel({'data': image_batch}, is_training=args.is_training,
                                               num_classes=args.num_classes, embedding_size=args.embedding_size,
                                               ASPP=args.ASPP, CRN=args.CRN)
    val_net = DeepLabResNetStructuredLearningModel({'data': val_image_batch}, is_training=False, reuse=True,
                                               num_classes=args.num_classes, embedding_size=args.embedding_size,
                                               ASPP=args.ASPP, CRN=args.CRN)
    # Predictions: ignoring all predictions with labels greater or equal than n_classes
    raw_output = net.get_raw_output()
    # Pixel-wise softmax loss.
    loss = net.get_loss(raw_output, label_batch)
    l2_losses = [args.weight_decay * tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'weights' in v.name]
    reduced_loss = tf.reduce_mean(loss) + tf.add_n(l2_losses)
    val_raw_output = val_net.get_raw_output()
    val_loss = val_net.get_loss(val_raw_output, val_label_batch)


    # Which variables to load. Running means and variances are not trainable,
    # thus all_variables() should be restored.
    restore_var = [v for v in tf.global_variables() if 'crn' not in v.name and ('fc' not in v.name or not args.not_restore_last) ]
    all_trainable = [v for v in tf.trainable_variables() if 'beta' not in v.name and 'gamma' not in v.name]
    fc_trainable = [v for v in all_trainable if 'fc' in v.name]
    conv_trainable = [v for v in all_trainable if 'fc' not in v.name]  # lr * 1.0
    fc_w_trainable = [v for v in fc_trainable if 'weights' in v.name]  # lr * 10.0
    fc_b_trainable = [v for v in fc_trainable if 'biases' in v.name]  # lr * 20.0
    crn_trainable = [v for v in tf.trainable_variables() if 'crn' in v.name]
    assert (len(all_trainable) == len(fc_trainable) + len(conv_trainable))
    assert (len(fc_trainable) == len(fc_w_trainable) + len(fc_b_trainable))

    # Processed predictions: for visualisation.
    raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(image_batch)[1:3, ])
    raw_output_up = tf.argmax(raw_output_up, dimension=3)
    pred = tf.expand_dims(raw_output_up, dim=3)

    # Define loss and optimisation parameters.
    base_lr = tf.constant(args.learning_rate)
    step_ph = tf.placeholder(dtype=tf.float32, shape=())
    learning_rate = tf.scalar_mul(base_lr, tf.pow((1 - step_ph / args.num_steps), args.power))

    opt_conv = tf.train.MomentumOptimizer(learning_rate, args.momentum)
    opt_fc_w = tf.train.MomentumOptimizer(learning_rate * 10.0, args.momentum)
    opt_fc_b = tf.train.MomentumOptimizer(learning_rate * 20.0, args.momentum)
    opt_crn = tf.train.MomentumOptimizer(learning_rate * 10.0, args.momentum)

    grads = tf.gradients(reduced_loss, conv_trainable + fc_w_trainable + fc_b_trainable + crn_trainable)
    grads_conv = grads[:len(conv_trainable)]
    grads_fc_w = grads[len(conv_trainable): (len(conv_trainable) + len(fc_w_trainable))]
    grads_fc_b = grads[(len(conv_trainable) + len(fc_w_trainable)): (len(conv_trainable) + len(fc_w_trainable) + len(fc_b_trainable))]
    grads_opt_crn = grads[(len(conv_trainable) + len(fc_w_trainable) + len(fc_b_trainable)):]

    train_op_conv = opt_conv.apply_gradients(zip(grads_conv, conv_trainable))
    train_op_fc_w = opt_fc_w.apply_gradients(zip(grads_fc_w, fc_w_trainable))
    train_op_fc_b = opt_fc_b.apply_gradients(zip(grads_fc_b, fc_b_trainable))
    train_op_crn = opt_crn.apply_gradients(zip(grads_opt_crn, crn_trainable))

    train_op = tf.group(train_op_conv, train_op_fc_w, train_op_fc_b, train_op_crn)


    # Image summary.
    images_summary = tf.py_func(inv_preprocess, [image_batch, args.save_num_images, IMG_MEAN], tf.uint8)
    labels_summary = tf.py_func(decode_labels, [label_batch, args.save_num_images, args.num_classes], tf.uint8)
    preds_summary = tf.py_func(decode_labels, [pred, args.save_num_images, args.num_classes], tf.uint8)
    tf.summary.image('images',
                     tf.concat(axis=2, values=[images_summary, labels_summary, preds_summary]),
                     max_outputs=args.save_num_images)  # Concatenate row-wise.
    # Loss summary.
    tf.summary.scalar('TRAIN/loss', tf.reduce_mean(loss))
    tf.summary.scalar('TRAIN/loss_l2', reduced_loss)
    tf.summary.scalar('TRAIN/lr', learning_rate)
    # Tensor summary.
    tensor_to_summary = net.tensor_to_summary
    tensor_to_summary['raw_outputs'] = raw_output
    for key, t in tensor_to_summary.items():
        tf.summary.histogram('SCORE/' + t.op.name + '/' + key + '/scores', t)
    # Activation summary.
    variables_to_summary = fc_w_trainable + fc_b_trainable + crn_trainable
    for v in variables_to_summary:
        tf.summary.histogram('ACT/' + v.op.name + '/activations', v)
        tf.summary.scalar('ACT/' + v.op.name + '/zero_fraction',
                      tf.nn.zero_fraction(v))
    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(args.snapshot_dir,
                                           graph=tf.get_default_graph())

    # validation summary, basic just loss summary
    val_snapshot_dir = args.snapshot_dir + '/val'
    if not os.path.exists(val_snapshot_dir):
      os.makedirs(val_snapshot_dir)
    val_summaries = [tf.summary.scalar('TRAIN/loss', tf.reduce_mean(val_loss))]
    val_summary_op = tf.summary.merge(val_summaries)
    val_summary_writer = tf.summary.FileWriter(val_snapshot_dir)


    # Set up tf session and initialize variables.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()

    sess.run(init)

    # Saver for storing checkpoints of the model.
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=10)

    # Load variables if the checkpoint is provided.
    if args.restore_from is not None:
        loader = tf.train.Saver(var_list=restore_var)
        load(loader, sess, args.restore_from)

    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    # Iterate over training steps.
    for step in range(args.num_steps):
        start_time = time.time()
        feed_dict = {step_ph: step}

        if step % args.summary_every == 0:
            loss_value, images, labels, preds, summary, _ = sess.run(
                [reduced_loss, image_batch, label_batch, pred, summary_op, train_op], feed_dict=feed_dict)
            summary_writer.add_summary(summary, step)

            _, _, val_summary = sess.run([val_image_batch, val_label_batch, val_summary_op], feed_dict=feed_dict)
            val_summary_writer.add_summary(val_summary, step)
        else:
            loss_value, _ = sess.run([reduced_loss, train_op], feed_dict=feed_dict)
        duration = time.time() - start_time

        if step % args.save_pred_every ==0:
            save(saver, sess, args.snapshot_dir, step)
        print('step {:d} \t loss = {:.3f}, ({:.3f} sec/step)'.format(step, loss_value, duration))

    coord.request_stop()
    coord.join(threads)


if __name__ == '__main__':
    main()
