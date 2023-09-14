import os 
import cv2
from skimage import io, color
import numpy as np
from torchvision import transforms
import albumentations as A
import PIL.Image as Image
class BackGround(object):
    """Operator that resizes to the desired size while maintaining the ratio
            fills the remaining part with a black background

        Args:
            output_size (tuple or int): Desired output size. If tuple, output is
                matched to output_size.
    """
    
    def __init__(self, output_size, in_channel=1):
        self.output_size = output_size
        self.in_channel = in_channel
    def __call__(self, image):
        h, w = image.shape[:2]

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size, self.output_size * w / h
            else:
                new_h, new_w = self.output_size * h / w, self.output_size
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        
        img = cv2.resize(image, (new_w, new_h))
        if self.in_channel ==1:
            new_image = np.zeros((self.output_size, self.output_size))
            if h > w:
                new_image[:,(self.output_size//2 - new_w//2):(self.output_size//2 - new_w//2 + new_w)] = img
            else:
                new_image[(self.output_size//2 - new_h//2):(self.output_size//2 - new_h//2 + new_h), :] = img

            return new_image
        else:
            new_image = np.zeros((self.output_size, self.output_size,3))
            if h > w:
                new_image[:,(self.output_size//2 - new_w//2):(self.output_size//2 - new_w//2 + new_w),:] = img
            else:
                new_image[(self.output_size//2 - new_h//2):(self.output_size//2 - new_h//2 + new_h), :,:] = img

            return new_image
        
        
class CustomDataset():
    def __init__(self, df, transform=True, mode="OCT"):
        self.df = df
        if mode == "OCT":
            self.background1 = BackGround(256)
            self.background2 = BackGround(256)
        elif mode =="brain":
            self.background1 = BackGround(256, 3)
            self.background2 = BackGround(256)
            self.kind = ["CS", "DU", "EZ", "FG", "HT"]
        self.to_tensor = transforms.ToTensor()
        self.normalize = lambda x: x/255
        # for vis
        # self.unnormalize = transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        #                                         std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
        self.to_pil = transforms.ToPILImage()
        self.transformed = A.Compose([
            A.HorizontalFlip(p=0.3),
            A.GaussianBlur((3,7), p=0.3)
        ])
        self.transform = transform
        self.mode = mode
    def __getitem__(self, i):
        sample = self.df.iloc[i]
        if self.mode == "OCT":
            image = cv2.imread(sample["path"], cv2.IMREAD_GRAYSCALE)
            label = cv2.imread(sample["class"], cv2.IMREAD_GRAYSCALE)
            
            image_ = self.background1(image)
            label_ = self.background2(label)
            
            if self.transform:
                transformed = self.transformed(image=image_, label=label_)
                image_ = transformed["image"]
                label_ = transformed["label"]
            
            ret={}
            image_= self.to_tensor(image_)
            label_= self.to_tensor(label_)
            
            
            image_ = self.normalize(image_)
            label_ = self.normalize(label_)
            
            ret["id"] = sample["path"].split("/")[-1]
            ret["image"] = image_
            ret["label"] = label_
            ret["pilimage"] = self.to_pil(image_.squeeze())
            return ret
        elif self.mode == "brain":
            image = cv2.imread(sample["path"], cv2.IMREAD_COLOR)
            label = cv2.imread(sample["class"], cv2.IMREAD_GRAYSCALE)
            
            image_ = self.background1(image)
            label_ = self.background2(label)
            
            if self.transform:
                transformed = self.transformed(image=image_, label=label_)
                image_ = transformed["image"]
                label_ = transformed["label"]
            
            ret={}
            image_= self.to_tensor(image_)
            label_= self.to_tensor(label_)
            
            
            image_ = self.normalize(image_)
            label_ = self.normalize(label_)
            
            ret["id"] = sample["path"].split("/")[-1]
            ret["class"] = [int(i) for i, v in enumerate(self.kind) if v in ret["id"]]
            ret["image"] = image_
            ret["label"] = label_
            ret["pilimage"] = self.to_pil(image_.squeeze())
            return ret
    def __len__(self):
        return len(self.df)