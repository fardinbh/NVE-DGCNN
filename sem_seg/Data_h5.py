import os
import numpy as np
from collections import Counter

# Utils
import data_prep_util
import indoor3d_util

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
NUM_POINT = 8192
H5_BATCH_SIZE = 4000
DATA_DIM = [NUM_POINT, 10]
LABEL_DIM = [NUM_POINT]
DATA_DTYPE = 'float32'
LABEL_DTYPE = 'uint8'


class PathManager:
    def __init__(self, base_dir):
        self.root_dir = os.path.dirname(base_dir)
        self.data_dir = os.path.join(self.root_dir, 'data')
        self.indoor3d_data_dir = os.path.join(self.data_dir, 'bridge_npy')
        self.filelist = os.path.join(base_dir, 'meta/all_data_label.txt')
        self.data_label_files = [os.path.join(self.indoor3d_data_dir, line.rstrip()) for line in open(self.filelist)]
        self.output_dir = os.path.join(self.data_dir, 'bridge_npy_hdf5_data')
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        self.output_filename_prefix = os.path.join(self.output_dir, 'ply_data_all')
        self.output_room_filelist = os.path.join(self.output_dir, 'room_filelist.txt')


class HDF5BatchManager:
    def __init__(self, path_manager):
        self.batch_data_dim = [H5_BATCH_SIZE] + DATA_DIM
        self.batch_label_dim = [H5_BATCH_SIZE] + LABEL_DIM
        self.h5_batch_data = np.zeros(self.batch_data_dim, dtype=np.float32)
        self.h5_batch_label = np.zeros(self.batch_label_dim, dtype=np.uint8)
        self.buffer_size = 0
        self.h5_index = 0
        self.path_manager = path_manager

    def insert_batch(self, data, label, last_batch=False):
        data_size = data.shape[0]
        if self.buffer_size + data_size <= self.h5_batch_data.shape[0]:
            self._fit_data_into_buffer(data, label, data_size)
        else:
            self._store_and_recall(data, label, data_size)

        if last_batch and self.buffer_size > 0:
            self._store_current_buffer()

    def _fit_data_into_buffer(self, data, label, data_size):
        self.h5_batch_data[self.buffer_size:self.buffer_size + data_size, ...] = data
        self.h5_batch_label[self.buffer_size:self.buffer_size + data_size] = label
        self.buffer_size += data_size

    def _store_and_recall(self, data, label, data_size):
        capacity = self.h5_batch_data.shape[0] - self.buffer_size
        if capacity > 0:
            self._fit_data_into_buffer(data[:capacity, ...], label[:capacity, ...], capacity)
        self._store_current_buffer()
        self.insert_batch(data[capacity:, ...], label[capacity:, ...])

    def _store_current_buffer(self):
        h5_filename = self.path_manager.output_filename_prefix + '_' + str(self.h5_index) + '.h5'
        data_prep_util.save_h5(h5_filename, self.h5_batch_data, self.h5_batch_label, DATA_DTYPE, LABEL_DTYPE)
        print('Stored {0} with size {1}'.format(h5_filename, self.buffer_size))
        self.h5_index += 1
        self.buffer_size = 0


def process_data_files(path_manager):
    h5_manager = HDF5BatchManager(path_manager)
    sample_cnt = 0

    with open(path_manager.output_room_filelist, 'w') as fout_room:
        for i, data_label_filename in enumerate(path_manager.data_label_files):
            print(data_label_filename)
            data, label = indoor3d_util.room2blocks_wrapper_normalized(data_label_filename, NUM_POINT,
                                                                         block_size=0.40, stride=0.30,
                                                                         random_sample=False, sample_num=None)
            print('{0}, {1}'.format(data.shape, label.shape))
            for _ in range(data.shape[0]):
                fout_room.write(os.path.basename(data_label_filename)[0:-4] + '\n')
            sample_cnt += data.shape[0]
            h5_manager.insert_batch(data, label, i == len(path_manager.data_label_files) - 1)

    print("Total samples: {0}".format(sample_cnt))


if __name__ == "__main__":
    path_manager = PathManager(BASE_DIR)
    process_data_files(path_manager)
