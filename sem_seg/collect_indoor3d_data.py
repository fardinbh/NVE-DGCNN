import os
import sys

# Constants
BASE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
ROOT_DIRECTORY = os.path.dirname(BASE_DIRECTORY)
ANNOTATION_PATHS_FILE = os.path.join(BASE_DIRECTORY, 'meta/anno_paths.txt')
OUTPUT_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'data/bridge')

# Append the base directory to system path for imports
sys.path.append(BASE_DIRECTORY)

# Assuming indoor3d_util has the DATA_PATH and collect_point_label defined
import indoor3d_util


def fetch_annotation_paths() -> list:
    """
    Retrieves the annotation paths from the provided file.
    
    Returns:
        List of annotation paths.
    """
    with open(ANNOTATION_PATHS_FILE, 'r') as file:
        raw_paths = file.readlines()
        
    return [os.path.join(indoor3d_util.DATA_PATH, path.strip()) for path in raw_paths]


def ensure_directory(directory_path: str):
    """
    Checks and creates a directory if it doesn't exist.

    Args:
        directory_path: Path of the directory to check/create.
    """
    if not os.path.exists(directory_path):
        os.mkdir(directory_path)


def process_annotation_paths(anno_paths: list):
    """
    Processes the list of annotation paths and invokes the utility function to collect point labels.

    Args:
        anno_paths: List of annotation paths.
    """
    for path in anno_paths:
        print(path)
        try:
            path_elements = path.split('/')
            output_filename = f"{path_elements[-3]}_{path_elements[-2]}.npy"
            indoor3d_util.collect_point_label(path, os.path.join(OUTPUT_DIRECTORY, output_filename), 'numpy')
        except Exception as e:
            print(f"Error processing path {path}: {str(e)}")


if __name__ == "__main__":
    ensure_directory(OUTPUT_DIRECTORY)
    annotation_paths = fetch_annotation_paths()
    process_annotation_paths(annotation_paths)

