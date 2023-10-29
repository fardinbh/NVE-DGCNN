import os
import numpy as np
import glob

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(ROOT_DIR, 'data')
MIRRORED_DIR = os.path.join(DATA_DIR, 'mirrored')
BRIDGE_DIR = os.path.join(DATA_DIR, 'bridge')
ANNO_PATHS = [line.rstrip() for line in open(os.path.join(BASE_DIR, 'meta/anno_paths_.txt'))]

def ensure_dir_exists(directory):
    """Ensure a directory exists, and if not, create it."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def mirror_coordinates(input_data):
    """Mirror the x-coordinates of the input data."""
    mirrored_data = np.copy(input_data)
    mirrored_data[:, 0] = -input_data[:, 0]
    return mirrored_data

def process_annotation(anno_path):
    """Process each annotation path."""
    elements = anno_path.split('/')
    area = os.path.join(MIRRORED_DIR, elements[-3])
    part = os.path.join(area, elements[-2])
    output_dir = os.path.join(part, 'Annotations')
    
    ensure_dir_exists(area)
    ensure_dir_exists(part)
    ensure_dir_exists(output_dir)

    input_file = os.path.join(BRIDGE_DIR, elements[-3], elements[-2], f"{elements[-2]}.txt")
    input_data = np.loadtxt(input_file, dtype=np.float, delimiter=' ')
    mirrored_data = mirror_coordinates(input_data)
    np.savetxt(os.path.join(part, f"{elements[-2]}.txt"), mirrored_data, fmt='%.3f %.3f %.3f %d %d %d')
    
    for f in glob.glob(os.path.join(BRIDGE_DIR, anno_path, '*.txt')):
        input_data = np.loadtxt(f, dtype=np.float, delimiter=' ')
        mirrored_data = mirror_coordinates(input_data)
        output_filename = os.path.basename(f)
        np.savetxt(os.path.join(output_dir, output_filename), mirrored_data, fmt='%.3f %.3f %.3f %d %d %d')

def main():
    """Main function to process all annotations."""
    ensure_dir_exists(MIRRORED_DIR)
    for anno_path in ANNO_PATHS:
        print(f"Processing: {anno_path}")
        process_annotation(anno_path)

if __name__ == "__main__":
    main()