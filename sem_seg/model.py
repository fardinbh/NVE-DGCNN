import math
import time
import os
import sys
import tensorflow as tf
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, '../models'))
import tf_util


class PointCloudModel:
    """
    A model to process point clouds.
    """

    def __init__(self, num_features=10, k_neighbors=20):
        self.num_features = num_features
        self.k_neighbors = k_neighbors

    def placeholder_inputs(self, batch_size, num_point):
        """
        Generate placeholder variables for point clouds and labels.
        
        Parameters:
        - batch_size: size of each batch
        - num_point: number of points in point cloud

        Returns:
        - pointclouds_pl: Placeholder for point cloud inputs
        - labels_pl: Placeholder for labels
        """
        pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, self.num_features))
        labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
        return pointclouds_pl, labels_pl

    def _edge_feature_extraction(self, net_input, k, scope_name, is_training, bn_decay):
        """
        Extract edge features from the input tensor using adjacency information.

        Parameters:
        - net_input: Input tensor
        - k: Number of neighbors
        - scope_name: Name scope for convolution operations
        - is_training: Flag for training mode
        - bn_decay: Batch normalization decay factor

        Returns:
        - net_output: Output tensor after edge feature extraction
        """
        adj = tf_util.pairwise_distance(net_input[:, :, 0:3])  # Using original XYZ
        nn_idx = tf_util.knn(adj, k=k)
        edge_feature = tf_util.get_edge_feature(net_input, nn_idx=nn_idx, k=k)
        net_output = tf_util.conv2d(edge_feature, 64, [1, 1], padding='VALID', stride=[1, 1],
                                    bn=True, is_training=is_training, scope=scope_name, bn_decay=bn_decay, is_dist=True)
        return net_output

    def get_model(self, point_cloud, is_training, bn_decay=None):
        """
        Construct the model to process point clouds.

        Parameters:
        - point_cloud: Input point cloud tensor
        - is_training: Flag for training mode
        - bn_decay: Batch normalization decay factor

        Returns:
        - net: Output tensor after processing
        """
        batch_size = point_cloud.get_shape()[0].value
        num_point = point_cloud.get_shape()[1].value
        input_image = tf.expand_dims(point_cloud, -1)

        net_1 = self._edge_feature_extraction(input_image, self.k_neighbors, 'adj_conv1', is_training, bn_decay)
        net_1 = self._edge_feature_extraction(net_1, self.k_neighbors, 'adj_conv2', is_training, bn_decay)
        net_1 = tf.reduce_max(net_1, axis=-2, keep_dims=True)

        net_2 = self._edge_feature_extraction(net_1, self.k_neighbors, 'adj_conv3', is_training, bn_decay)
        net_2 = self._edge_feature_extraction(net_2, self.k_neighbors, 'adj_conv4', is_training, bn_decay)
        net_2 = tf.reduce_max(net_2, axis=-2, keep_dims=True)

        net_3 = self._edge_feature_extraction(net_2, self.k_neighbors, 'adj_conv5', is_training, bn_decay)
        net_3 = tf.reduce_max(net_3, axis=-2, keep_dims=True)

        out7 = tf_util.conv2d(tf.concat([net_1, net_2, net_3], axis=-1), 1024, [1, 1],
                              padding='VALID', stride=[1, 1], bn=True, is_training=is_training, scope='adj_conv7',
                              bn_decay=bn_decay, is_dist=True)

        out_max = tf_util.max_pool2d(out7, [num_point, 1], padding='VALID', scope='maxpool')
        expand = tf.tile(out_max, [1, num_point, 1, 1])
        concat = tf.concat(axis=3, values=[expand, net_1, net_2, net_3])

        # Convolution operations
        net = tf_util.conv2d(concat, 512, [1, 1], padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training, scope='seg/conv1', is_dist=True)
        net = tf_util.conv2d(net, 256, [1, 1], padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training, scope='seg/conv2', is_dist=True)
        net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training, scope='dp1')
        net = tf_util.conv2d(net, 3, [1, 1], padding='VALID', stride=[1, 1],
                             activation_fn=None, scope='seg/conv3', is_dist=True)
        net = tf.squeeze(net, [2])

        return net

    @staticmethod
    def get_loss(pred, label):
        """
        Compute the loss between predictions and labels.

        Parameters:
        - pred: Predictions tensor
        - label: Labels tensor

        Returns:
        - Loss value
        """
        class_weights = tf.constant([0.714, 0.271, 0.016])  # Dataset-specific weights
        weights = tf.gather(class_weights, label)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=label, logits=pred, weights=weights)
        return tf.reduce_mean(loss)
