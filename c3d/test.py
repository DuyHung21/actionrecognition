import os
import json
import time
import argparse
from pathlib import PurePath

import numpy as np
import cv2
import torch
from torchvision import transforms

from utils.ucf.transform import Rescale, CenterCrop, ToTest
import config


def main(model_path, video_path):
    # Use cuda if it is possible
    location = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(location)
    print("Device being used:", device)

    with open(config.training['category_filepath']) as f:
        categories = json.load(f)
    
    model = torch.load(model_path, map_location=device)
    model.to(device)
    model.eval()

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(os.path.join(config.testing['output_folder'], 'result_{}.avi'.format(PurePath(video_path).name)),
                        fourcc, 
                        20.0, 
                        (int(cap.get(3)), int(cap.get(4))),
                        )

    test_transform = transforms.Compose([
        Rescale((171, 128)),
        CenterCrop((112, 112)),
        ToTest()
    ])

    category = 'unknown'
    probability = 0
    num_frame = 0
    clip = []
    while True:
        ret, frame = cap.read()
        if ret is None:
            break

        num_frame += 1
        clip.append(frame)
           
        start_time = time.time()
        if num_frame >= 16:
            clip_tensor = test_transform(np.array(clip))
            clip_tensor.to(device)

            output = model(clip_tensor)
            probs = torch.nn.Softmax(dim=1)(output)
            label = torch.max(probs, 1)[1].detach().cpu().numpy()[0]
            category = categories[label]
            probability = probs[0][label]
            clip.pop(0)

        run_time = time.time() - start_time
        cv2.putText(frame, category, (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255), 2)
        cv2.putText(frame, "prob: %.4f" % probability, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255), 2)

        cv2.putText(frame, "fps: %.4f" % (1 / run_time), (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255), 2)

        out.write(frame)
        cv2.imshow('result', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    out.release()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Performing action recognition on videos')
    parser.add_argument('--model', help='Location of model file')
    parser.add_argument('--video', help='Location of video file')

    args = parser.parse_args()
    main(args.model, args.video)