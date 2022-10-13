import torch
import clip
from PIL import Image
import os
import numpy as np
import cv2
import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
dataset_path = "/media/ubuntu/E/audio_video/yowo_dataset/text"
output_path = "/media/ubuntu/E/audio_video/yowo_dataset/video_embed"

video_name_list = os.listdir(dataset_path)


with torch.no_grad():
    # image_features = model.encode_image(image)
    # print(image_features.shape)

    for video_name in tqdm.tqdm(video_name_list):
        drop_mp4 = video_name.split(".")[0]
        video_path = os.path.join(dataset_path, video_name)
        cap = cv2.VideoCapture(video_path)
        save_feature = []
        while (cap.isOpened()):
            ret, frame = cap.read()

            if ret:
                frame = preprocess(Image.fromarray(frame)).unsqueeze(0).to(device)
                image_features = model.encode_image(frame)
                save_feature.append(image_features)
            else:
                break
        array_result = np.array([t.detach().cpu().numpy() for t in save_feature]).squeeze()
        np.save(os.path.join(output_path, "{}.npy".format(drop_mp4)), array_result)


