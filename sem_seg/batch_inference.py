import os
import sys
import argparse
import numpy as np
import tensorflow as tf

from model import placeholder_inputs, get_model, get_loss
import indoor3d_util

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOG_DIR = "log1test"
META_DIR = "meta"
NUM_CLASSES = 3


def parse_arguments():
    parser = argparse.ArgumentParser(description="3D Bridge Parsing")
    parser.add_argument('--gpu', type=int, default=1, help='GPU to use')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training')
    parser.add_argument('--num_point', type=int, default=8192, help='Number of points')
    parser.add_argument('--model_path', default=os.path.join(LOG_DIR, 'eval/epoch_50.ckpt'), help='Model checkpoint path')
    parser.add_argument('--dump_dir', default=os.path.join(LOG_DIR, 'dump'), help='Dump folder path')
    parser.add_argument('--output_filelist', default=os.path.join(LOG_DIR, 'output_filelist.txt'), help='Output file list')
    parser.add_argument('--room_data_filelist', default=os.path.join(META_DIR, 'area5_data_label_m.txt'), help='Room data file list')
    parser.add_argument('--no_clutter', action='store_true', help='Ignore the clutter class')
    parser.add_argument('--visu', action='store_true', help='Output visualization OBJ file')

    return parser.parse_args()


def initialize_logging(dump_dir, flags):
    if not os.path.exists(dump_dir):
        os.mkdir(dump_dir)

    log_file = os.path.join(dump_dir, 'log_evaluate.txt')
    with open(log_file, 'w') as f:
        f.write(str(flags) + '\n')

    return log_file


def log(log_file, message):
    with open(log_file, 'a') as f:
        f.write(message + '\n')
    print(message)


def setup_tensorflow(gpu_index):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    return tf.Session(config=config)


def report_confusion_matrix(conf_matrix, log_file):
    class_names = ['Crack Defect', 'Spall Defect', 'No-Defect']
    log(log_file, "Confusion Matrix:")
    header = " " * 12 + " ".join(["{:<12}".format(name) for name in class_names])
    log(log_file, header)
    
    for i, row in enumerate(conf_matrix):
        row_str = "{:<12}:".format(class_names[i]) + " ".join(["{:<12}".format(item) for item in row])
        log(log_file, row_str)

    accuracies = []
    for i in range(NUM_CLASSES):
        accuracy = conf_matrix[i][i] / sum(conf_matrix[i])
        accuracies.append(accuracy)
        log(log_file, "{} Accuracy: {:.4f}".format(class_names[i], accuracy))
    
    overall_accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)
    log(log_file, "Overall Accuracy: {:.4f}".format(overall_accuracy))


def evaluation_pipeline(sess, ops, room_path_list, dump_dir, log_file, num_classes, flags):
    global_confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)

    for room_path in room_path_list:
        r_confusion_matrix = evaluate_room(sess, ops, room_path, dump_dir, num_classes, flags)
        global_confusion_matrix += r_confusion_matrix

    report_confusion_matrix(global_confusion_matrix, log_file)


def evaluate_room(sess, ops, room_path, dump_dir, num_classes, flags):
    # Placeholder setups
    pointclouds_pl = ops['pointclouds_pl']
    labels_pl = ops['labels_pl']
    is_training_pl = ops['is_training_pl']
    pred_softmax = ops['pred_softmax']

    # Load data
    current_data, current_label = load_room_data(room_path, flags.num_point)
    file_size = current_data.shape[0]
    num_batches = file_size // flags.batch_size

    r_confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)

    # Iterate through data batches
    for batch_idx in range(num_batches):
        start_idx = batch_idx * flags.batch_size
        end_idx = (batch_idx + 1) * flags.batch_size

        feed_dict = {
            pointclouds_pl: current_data[start_idx:end_idx, :, :],
            labels_pl: current_label[start_idx:end_idx],
            is_training_pl: False
        }

        pred_val = sess.run(pred_softmax, feed_dict=feed_dict)
        pred_label = np.argmax(pred_val, 2)

        # Update confusion matrix
        for i in range(pred_label.shape[0]):
            for j in range(pred_label.shape[1]):
                gt_class = current_label[start_idx + i, j]
                pred_class = pred_label[i, j]
                r_confusion_matrix[gt_class, pred_class] += 1

    return r_confusion_matrix


def main():
    flags = parse_arguments()
    log_file = initialize_logging(flags.dump_dir, flags)
    room_path_list = [os.path.join(ROOT_DIR, line.strip()) for line in open(flags.room_data_filelist)]

    with tf.Graph().as_default():
        with setup_tensorflow(flags.gpu) as sess:
            # Placeholder setups
            pointclouds_pl, labels_pl = placeholder_inputs(flags.batch_size, flags.num_point)
            is_training_pl = tf.placeholder(tf.bool, shape=())

            pred = get_model(pointclouds_pl, is_training_pl)
            pred_softmax = tf.nn.softmax(pred)

            # Define necessary operations
            ops = {
                'pointclouds_pl': pointclouds_pl,
                'labels_pl': labels_pl,
                'is_training_pl': is_training_pl,
                'pred_softmax': pred_softmax,
            }

            saver = tf.train.Saver()
            saver.restore(sess, flags.model_path)
            log(log_file, "Model restored.")

            evaluation_pipeline(sess, ops, room_path_list, flags.dump_dir, log_file, NUM_CLASSES, flags)

    log(log_file, "Evaluation finished.")


if __name__ == "__main__":
    main()
