import os
import random
import json

import cv2
import numpy as np

UCF50_DATASET = 'UCF50'

def get_filename_and_classes(dataset_dir=UCF50_DATASET):
    """
        input: the directory of UCF50 dataset
        output:
            + files: list of all filepaths of files in the folder
            + classes: the corresponding class of each file
            + categories: list of unique category, of which id is the class

    """

    files = []
    classes = []
    categories = os.listdir(dataset_dir)

    for index, category in enumerate(categories):
        for filename in os.listdir(os.path.join(dataset_dir, category)):

            # Skipping files that are not movies
            if not filename.endswith('.avi'):
                continue

            files.append(os.path.join(os.path.abspath(dataset_dir), category, filename))
            classes.append(index)

    return files, classes, categories

def extract_random_clips(video_path, num_clips=5, frame_per_clip=16, frame_size=(171,128)):
    """
        input: 
            + video_path: the path of video file
            + num_clips: number of random clips extracted from video file
            + num_frames: number of frames per clip
            + frame_size: the size of each frame
        output: data extracted from video, used for training and testing
    """

    # Read video from file and collect frames
    cap = cv2.VideoCapture(video_path)
    frames = []

    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break 

        frames.append(cv2.resize(frame, frame_size))

    """
        Randomly extract clips from video frames.
        The clips should not have overlapped frames.
    """
    num_frames = len(frames)
    clips = []

    assert num_frames // frame_per_clip >= num_clips, "Not enough frames for extracting clips"

    while len(clips) < num_clips:
        frame_start = random.randint(0, num_frames)
        valid_frame_start = True

        # The clip must have {frame_per_clip} frames
        if frame_start + frame_per_clip >= num_frames:
            continue

        for c in clips:
            # The clip must not overlap with other clips in the array
            if (frame_start < c and frame_start + frame_per_clip > c) or \
                (frame_start > c and c + frame_per_clip > frame_start):
                valid_frame_start = False

        if not valid_frame_start:
            continue
        
        clips.append(frame_start)
    
    for i, clip_start in enumerate(clips):
        clips[i] = frames[clip_start:clip_start+frame_per_clip]

    return np.array(clips, dtype=np.uint8)


def extract_clip(clip_path):
    """
        input: 
            + clip_path: {file_path}_{clip_id}_{frame_per_clip}
            + file_path: the path of video file
            + clip_id: id of clip in the video file
            + frame_per_clip: number of frames in a clip
        output:
            + the extracted clip
    """

    # Extract relavent information from clip_path
    fragments = clip_path.split('_')
    video_path, clip_id, frame_per_clip = "_".join(fragments[0:-2]), int(fragments[-2]), int(fragments[-1])

    cap = cv2.VideoCapture(video_path)
    clip = []

    start_id = clip_id * frame_per_clip
    end_id = start_id + frame_per_clip

    i = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break 

        if i >= start_id and i < end_id:
            clip.append(frame)
        elif i > end_id:
            break

        i += 1

    return np.array(clip)


def count_clips(video_path, frame_per_clip=16):
    """
        input: 
            + video_path: the path of video file
            + frame_per_clip: number of frames per clip
        output:
            + num_frame: number of frames in the video
            + num_clips: number of valid clips that could be extracted
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break 

        frames.append(frame)

    num_frames = len(frames)
    return num_frames, num_frames // frame_per_clip


def get_clip_and_classes(files=None, file_classes=None, frame_per_clip=None, dataset_file=None):
        """
            input:
                + files: list of file path
                + file_classes: list of corresponding class of each file
                + frame_per_clip
                + dataset_file: get clip and classes from a pre-defined file

            output:
                + clips: 
                    * list of clips and files that contain that clip
                    * format: '{filename}_{clipId}'.
                    * filename: original file
                    * clipId: the index of clip in the file (for extracting)
                + clip_classes: the corresponding class of each clip
        """

        assert not ((files is None) and (file_classes is None) and \
            (frame_per_clip is None) and (dataset_file is None)),\
            "Please specify something"

        clips = []
        clip_classes = []
        categories = []

        if dataset_file is None:
            for f, fc in zip(files, file_classes):
                n_frame, n_clip = count_clips(f, frame_per_clip)
                for i in range(n_clip):
                    clips.append('{}_{}'.format(f, i))
                    clip_classes.append(fc)
        else:
            with open(dataset_file) as f:
                lines = f.readlines()
                for l in lines:
                    clip, category = l.replace("\n", "").split(" ")
                    clips.append(clip)
                    clip_classes.append(category)

            categories = list(set(clip_classes))
            for i in range(len(clip_classes)):
                for j, category in enumerate(categories):
                    if clip_classes[i] == category:
                        clip_classes[i] = j 
                        break

        return clips, clip_classes, categories


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Prepare clip dataset text file for UCF dataset')
    parser.add_argument('-c', help='Create a text file for the dataset')
    parser.add_argument('-d', help='Dataset directory', required=True)
    parser.add_argument('-p', help='Path of the text file for reading and for creating', required=True)

    args = parser.parse_args()

    if args.c is not None:
        files, classes, categories = get_filename_and_classes(dataset_dir=args.d)

        print(files[0])

        clips, clip_classes, _ = get_clip_and_classes(files, classes, 16)
        print(clips[0], len(clips))

        content = ['{} {}\n'.format(clip, class_) for clip, class_ in zip(clips, clip_classes)]
        
        with open(args.p, 'w+') as f:
            f.writelines(content)

        with open(args.p.replace('txt', 'json'), 'w') as f:
            json.dumps(f, categories)

    else:
        clips, clip_classes, test_categories = get_clip_and_classes(dataset_file=args.p)
        with open(args.p.replace('txt', 'json'), 'w') as f:
            json.dump(test_categories, f)

    b = False
    for j, c in enumerate(clips):
        clip = extract_clip(c)
        for i in range(clip.shape[0]):
            cv2.imshow(str(j), clip[i,:,:,:])
            if cv2.waitKey(0) & 0xFF == ord('q'):
                b = True
                break
        if b:
            break
    
    cv2.destroyAllWindows()