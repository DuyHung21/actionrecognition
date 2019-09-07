import os
import argparse
import subprocess
import json
from pathlib import Path, PurePath

import numpy as np
import cv2


def count_frame(video_path):
    """Read a video file and count the number of frames in 
    that file
    """

    cap = cv2.VideoCapture(video_path)
    num_frames = 0
    while True:
        (grabbed, frame) = cap.read()

        if not grabbed:
            break

        num_frames += 1

    return num_frames


def get_files_info(directory):
    """Retrive all filepath as well as their information in a directory
    
    Loop through the dataset directory, read all video files and retrive 
    the path, the number of frame and the category of each file

    Args:
        directory (string): path to dataset directory

    Returns:
        A dict mapping each video filepath with the video's num_frame
        and category. For example:

        {"/directory/Skijet/001.avi": {
                                    "frames": 90,
                                    "class": "Skijet"
                                    },
        "/directory/Skijet/002.avi": {
                                    "frames": 80,
                                    "class": "Skijet"
                                    }                         
        }
    """
    assert isinstance(directory, str)
    directory = Path(directory)

    info = {}
    for category in directory.iterdir():
        if not category.is_dir():
            continue

        class_ = PurePath(category).name
        print("In folder:", class_)

        for file_ in category.iterdir():
            abs_path = str(file_.resolve())

            info[file_] = {
                        "frames": count_frame(abs_path),
                        "class": class_
            }
    
    return info


def main(dataset_directory, output_directory, frame_per_clip=16):
    """Access data directory, retrieve information and save results to output directory

    Iterate through all files in data directory, retrieve their relavent information, 
    including their absolute paths, categories and number of frames. The retrieved data
    is used to compute number of clips in the video, since the model actually work with 
    clips, not videos. The final result is saved to ouput directory.

    Args:
        dataset_directory (string): directory of the dataset,
        output_directory (string): directory where the results will be saved
        frame_per_clip (int): the number of each frames in an extracted clip.
            This parameter is used to generete the result file.

    Return:
        Save "dataset.txt" to output_directory. "dataset.txt" is a file that contains
            all information need for training. Each line in the file specifies the 
            information of a clip used for training. The information includes video 
            file path of the clip (1), index of the clip in the video (2), 
            number of frames in a clip (3) and the label to which that clip belongs (4).
            The information is separated by a space. For example:

            /dataset/UCF50/BaseballPitch/v_BaseballPitch_g03_c07.avi 3 16 BaseballPitch

        Save "category.json" to output_directory. "category.json" contains an array
            of all classes in the dataset.
    """

    print("Counting number of frames in each video")
    data = get_files_info(dataset_directory)
    print("Done")
    
    file_path = data.keys()
    frame_per_file = []
    classes = []
    
    for file_ in file_path:
        file_info = data[file_]
        frame_per_file.append(file_info['frames'])
        classes.append(file_info['class'])

    categories = list(set(classes))

    lines = []
    for file_, num_frame, class_ in zip(file_path,frame_per_file, classes):
        num_clip = num_frame // frame_per_clip

        label = categories.index(class_)
        for clip_id in range(num_clip):
            lines.append("{} {} {} {}\n".format(os.path.abspath(file_), 
                                                clip_id, 
                                                frame_per_clip, 
                                                label))

    with open(os.path.join(output_directory, 'dataset.txt'), 'w') as f:
        f.writelines(lines)

    with open(os.path.join(output_directory, 'category.txt'), 'w') as f:
        json.dump(categories, f)

    print('Done')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare dataset for video recognition')
    parser.add_argument('-d', help='dataset directory', required=True)
    parser.add_argument('-o', help='output directory', required=True)
    parser.add_argument('-nf', help='number of frame per clip', type=int, default=16)

    args = parser.parse_args()
    frame_per_clip = args.nf

    main(args.d, args.o, frame_per_clip)
    