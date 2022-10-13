import os
from PIL import Image
import numpy as np
import cv2
import torch
from torchvision import transforms
from networks.dan import DAN
import tqdm


class Model():
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data_transforms = transforms.Compose([
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
                                ])
        self.labels = ['neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger', 'contempt']
        # self.labels = ['neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger']

        self.model = DAN(num_head=4, num_class=8, pretrained=False)
        # affecnet8_epoch5_acc0.6209.pth rafdb_epoch21_acc0.897_bacc0.8275.pth
        checkpoint = torch.load('./checkpoints/affecnet8_epoch5_acc0.6209.pth',
            map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        self.model.to(self.device)
        self.model.eval()

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    
    def detect(self, img0):
        img = cv2.cvtColor(np.asarray(img0),cv2.COLOR_RGB2BGR)
        faces = self.face_cascade.detectMultiScale(img)
        
        return faces

    def fer(self, img0):

        # img0 = Image.open(path).convert('RGB')
        img0 = Image.fromarray(img0)

        faces = self.detect(img0)

        if len(faces) == 0:
            return 'null'

        ##  single face detection
        x, y, w, h = faces[0]

        img = img0.crop((x,y, x+w, y+h))

        img = self.data_transforms(img)
        img = img.view(1,3,224,224)
        img = img.to(self.device)

        with torch.set_grad_enabled(False):
            out, input_, head_feature = self.model(img)
            _, pred = torch.max(out,1)
            index = int(pred)
            label = self.labels[index]

            return label, head_feature.sum(dim=1)

if __name__ == "__main__":

    model = Model()

    dataset_path = "/media/ubuntu/E/audio_video/yowo_dataset/text"
    output_path = "/media/ubuntu/E/audio_video/yowo_dataset/video_embed"

    video_name_list = os.listdir(dataset_path)

    for video_name in tqdm.tqdm(video_name_list):
        drop_mp4 = video_name.split(".")[0]
        video_path = os.path.join(dataset_path, video_name)

        cap = cv2.VideoCapture(video_path)
        save_feature = []
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                try:
                    label, feature = model.fer(frame)
                    save_feature.append(feature)
                except:
                    pass
            else:
                break
        array_result = np.array([t.detach().cpu().numpy() for t in save_feature]).squeeze()
        np.save(os.path.join(output_path, "{}.npy".format(drop_mp4)), array_result)


