import os
import json
import time

import numpy as np
import cv2
import torch
from torchvision import transforms

from utils.ucf.transform_ucf50 import Rescale, CenterCrop, ToTest


location = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(location)
print("Device being used:", device)


if __name__ == "__main__":
    model_path = 'models_ucf50_final.pth' 
    test_video_path = 'test4.mp4'
    with open("./dataset/categories.json", "r") as read_file:
        categories = json.load(read_file)

    labels, label_names = categories.values(), categories.keys()
    labels, label_names = zip(*sorted(zip(labels, label_names)))
    print(label_names)



    model = torch.load(model_path, map_location=location)
    model.to(device)

    model.eval()   

    cap = cv2.VideoCapture(test_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    print(cap.get(3), cap.get(4))
    out = cv2.VideoWriter('output_test4.avi',fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    frame_id = 0
    frame_per_clip = 16

    clip = []

    test_transform = transforms.Compose([
        Rescale((171, 128)),
        CenterCrop((112, 112)),
        ToTest()
    ])

    category = 'unknown'
    while True:
        ret, frame = cap.read()

        if ret is None:
            break

        frame_id += 1
        clip.append(frame)

        if frame_id >= 16:
            clip_tensor = test_transform(np.array(clip))
            clip_tensor.to(device)

            start_time = time.time()
            output = model(clip_tensor)
            run_time = time.time() - start_time
            probs = torch.nn.Softmax(dim=1)(output)
            label = torch.max(probs, 1)[1].detach().cpu().numpy()[0]

            cv2.putText(frame, label_names[label], (20, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 2)
            cv2.putText(frame, "prob: %.4f" % probs[0][label], (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 2)

            cv2.putText(frame, "fps: %.4f" % (1 / run_time), (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 2)

            clip.pop(0)

        out.write(frame)
        cv2.imshow('result', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    out.release()


