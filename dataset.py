import os 
import cv2
from skimage import io, color
import numpy as np
from torchvision import transforms
import albumentations as A
import PIL.Image as Image
import SimpleITK as sitk
from skimage import morphology
import skimage.io as io
from sklearn.cluster import KMeans
from skimage.transform import resize
import torch
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
        
        img = resize(image, (new_w, new_h))
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
        

class BackGround2(object):
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
        
        img = cv2.resize(image, (self.output_size, self.output_size))
        
        return img
def Normalizing(data):
    maxi=torch.max(data)
    mini=torch.min(data)
    if maxi ==0:
        return data        
    result = (data-mini)/(maxi-mini)
    return result

class CustomDataset():
    def __init__(self, df, transform=True, mode="gray"):
        self.df = df
        if mode == "OCT":
            pass
        elif mode =="brain":
            self.background1 = BackGround(256, 3)
            self.background2 = BackGround(256)
            self.kind = ["CS", "DU", "EZ", "FG", "HT"]
            self.normalize = transforms.Normalize
        elif mode =="gray":
            self.background1 = BackGround(256, 3)
            self.background2 = BackGround(256)
            self.kind = ["CS", "DU", "EZ", "FG", "HT"]
            self.normalize = transforms.Normalize(0.5, 0.5)
        self.to_tensor = transforms.ToTensor()
        self.lambdaf = lambda x: x/255
        self.to_pil = transforms.ToPILImage()
        self.transformed = A.Compose([
            A.HorizontalFlip(p=0.3)
        ])
        
        self.transform = transform
        self.mode = mode
    def __getitem__(self, i):
        sample = self.df.iloc[i]
        ret ={}
        if self.mode == "gray":
            image = io.imread(sample["path"])
            label = io.imread(sample["class"])
        
            image = self.background2(image)
            label = self.background2(label)
            
            image = self.to_tensor(image)
            label = self.to_tensor(label)
            
            image = Normalizing(image)
            label = Normalizing(label)
            
            ret["id"] = sample["path"].split("/")[-1]
            
            ret["image"] = image
            ret["label"] = label
            
            return ret
        elif self.mode == "brain":
            image = cv2.imread(sample["path"], cv2.IMREAD_GRAYSCALE)
            label = cv2.imread(sample["class"], cv2.IMREAD_GRAYSCALE)
            
            
            image_ = self.background2(image)
            
            label_ = self.background2(label)
           
            if self.transform:
                transformed = self.transformed(image=image_, label=label_)
                image_ = transformed["image"]
                label_ = transformed["label"]
            
            # if len(image_[image_>102].shape)>0:
            ret={}
            image_= self.to_tensor(image_)
            label_= self.to_tensor(label_)
            image_std = image_.std(dim=(0,1,2))
            # label_std = label_.std(dim=(0,1,2))
            image_mean = image_.mean(dim=(0,1,2))
            # label_mean = label_.mean(dim=(0,1,2))
            try :
                image_norm = self.normalize(image_mean, image_std)
                image_ = image_norm(image_)
            except ValueError as e:
                image_norm = self.normalize(0.5, 0.5)
                image_ = image_norm(image_)
            
            ret["id"] = sample["path"].split("/")[-1]
            
            ret["image"] = image_
            ret["label"] = label_
                
            return ret
            # else:
            #     ret={}
            #     image_= self.to_tensor(image_)
            #     label_= self.to_tensor(np.zeros((256,256)))
                
            #     image_ = Normalizing(image_)
            #     # label_ = self.normalize(label_)
                
            #     ret["id"] = sample["path"].split("/")[-1]
                
            #     ret["image"] = image_
            #     ret["label"] = label_
                
            #     return ret
    def __len__(self):
        return len(self.df)
    
    


class CustomDataset2():
    def __init__(self, df, transform=True, mode="OCT"):
        self.df = df
        if mode == "OCT":
            pass
        elif mode =="brain":
            self.background1 = BackGround2(256)
            self.background2 = BackGround2(256)
            self.kind = ["CS", "DU", "EZ", "FG", "HT"]
        self.to_tensor = transforms.ToTensor()
        self.normalize = Normalizing

        self.to_pil = transforms.ToPILImage()
        self.transformed = A.Compose([
            A.HorizontalFlip(p=0.3)
        ])
        self.transform = transform
        self.mode = mode
    def __getitem__(self, i):
        
        sample = self.df.iloc[i]
        if self.mode == "OCT":
            pass
        elif self.mode == "brain":
            image = sitk.ReadImage(sample["path"], sitk.sitkFloat32)
            label = sitk.ReadImage(sample["class"], sitk.sitkUInt8)
            oriImage = sitk.GetArrayFromImage(image)
            image = N4BiasFieldCorrection(image)
            
            image = sitk.GetArrayFromImage(image)
            
            label = sitk.GetArrayFromImage(label)
            
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
            # ret["ori"]= oriImage
            ret["blend"]= image
            return ret
    def __len__(self):
        return len(self.df)


def N4BiasFieldCorrection(image:sitk.Image, numberOfIteration=1):
    transformed = sitk.RescaleIntensity(image, 0, 255)
    transformed = sitk.HuangThreshold(transformed,0,1)
    head_mask = transformed
    
    shrinkFactor = 4 #1
    inputImage = image

    inputImage = sitk.Shrink( image, [ shrinkFactor ] * inputImage.GetDimension() ) 
    maskImage = sitk.Shrink( head_mask, [ shrinkFactor ] * inputImage.GetDimension() ) 
    
    bias_corrector = sitk.N4BiasFieldCorrectionImageFilter() 

    corrected = bias_corrector.Execute(inputImage, maskImage) 
    
    log_bias_field = bias_corrector.GetLogBiasFieldAsImage(image) 
    corrected_image_full_resolution = sitk.Divide(image , sitk.Cast(sitk.Exp(log_bias_field),sitk.sitkFloat32))
    
    return corrected_image_full_resolution

def skullStriping(img):
  row_size = img.shape[0]
  col_size = img.shape[1]
  mean = np.mean(img)
  std = np.std(img)
  img = img - mean
  img = img / std
  middle = img[int(col_size / 5):int(col_size / 5 * 4), int(row_size / 5):int(row_size / 5 * 4)]
  mean = np.mean(middle)
  max = np.max(img)
  min = np.min(img)
  img[img == max] = mean
  img[img == min] = mean
  kmeans = KMeans(n_clusters=2).fit(np.reshape(middle, [np.prod(middle.shape), 1]))
  centers = sorted(kmeans.cluster_centers_.flatten())
  threshold = np.mean(centers)
  thresh_img = np.where(img < threshold, 1.0, 0.0)  # threshold the image
  eroded = morphology.erosion(thresh_img, np.ones([3, 3]))
  dilation = morphology.dilation(eroded, np.ones([5, 5]))
  eroded = morphology.erosion(dilation, np.ones([3, 3]))
  return eroded

def blended(img, mask):
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    img = img.astype(np.uint8)
    mask = (mask*255).astype(np.uint8)
    # mask = cv2.erode(mask, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    extracted = cv2.bitwise_and(img, img, mask=mask)
    notExtracted = cv2.bitwise_and(img,img, mask=(255-mask))
    return np.clip(extracted+notExtracted*0.7,0,255)

def skeletonizer(mask, method="zhang"):
    return morphology.skeletonize(mask, method=method)

def findCountourDrawPoly(image):
    zero = np.zeros_like(image, dtype="uint8")
    contours, _=cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    zero=cv2.polylines(zero, contours, True, 255, 1)
    zero = morphology.convex_hull_image(zero>0)
    return zero