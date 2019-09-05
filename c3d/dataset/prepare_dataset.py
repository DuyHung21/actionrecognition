import os
import argparse
import subprocess
import json

import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare dataset for video recognition')
    parser.add_argument('-d', help='dataset directory', required=True)
    parser.add_argument('-o', help='output directory', required=True)
    parser.add_argument('-nf', help='number of frame per clip', type=int, default=16)

    args = parser.parse_args()
    frame_per_clip = args.nf

    print("Counting number of frames in each video")
    # dataset_dir = args.d
    # if dataset_dir.endswith('/'):
    #     dataset_dir = dataset_dir[:-1]
    # subprocess.call(['./count_frame.sh', dataset_dir, 'temp.json'])
    print("Done")
    
    print("Generating text files")
    with open('temp.json') as f:
        data = json.load(f)
    
    file_path = data.keys()
    frame_per_file = []
    class_per_file = []
    
    for file_ in file_path:
        file_info = data[file_]
        frame_per_file.append(file_info['frames'])
        class_per_file.append(file_info['class'])

    categories = list(set(class_per_file))

    lines = []
    for file_, num_frame, class_ in zip(file_path,frame_per_file, class_per_file):
        num_clip = num_frame // frame_per_clip

        label = categories.index(class_)
        for clip_id in range(num_clip):
            lines.append("{} {} {} {}\n".format(os.path.abspath(file_), clip_id, frame_per_clip, label))

    with open(os.path.join(args.o, 'dataset.txt'), 'w') as f:
        f.writelines(lines)

    with open(os.path.join(args.o, 'category.txt'), 'w') as f:
        json.dump(categories, f)

    os.remove('temp.json')
    print('Done')

    
