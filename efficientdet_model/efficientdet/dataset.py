import os
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from torchvision import transforms
import cv2


class CocoDataset(Dataset):
    def __init__(self, root_dir, set='train2017', transform=None, policy_container=None, use_only_aug=False):
        '''
        Definition of the Dataset, policies and transformation to be used.

        Params
        :root_dir (string): root location of the dataset.
        :set (string): name of the dataset - this name must match the physical folder.
        :transform (torchvision.transforms.Compose): sequence of transformations to apply to the data.
        :policy_container (bbaug.policies): set of policies from where one will be chosen to be applied.
        :use_only_aug (bool) -> indicates if training will use only augmented images.
        '''
        self.root_dir = root_dir
        self.set_name = set
        self.transform = transform

        self.coco = COCO(os.path.join(self.root_dir, 'annotations', 'instances_' + self.set_name + '.json'))
        self.image_ids = self.coco.getImgIds()

        self.load_classes()
        #-----------------
        self.policy_container = policy_container
        self.to_tensor = transforms.ToTensor()
        self.use_only_aug = use_only_aug 
        #-----------------


    def load_classes(self):
        '''
        Load classes/categories from the dataset. Load the results to self-inner-class variables.

        Params
        :None

        Return
        :None
        '''
        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes = {}
        for c in categories:
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key


    def __len__(self):
        '''
        Get the total number of images in the dataset.
        
        Return
        :int -> total number of images.
        '''
        return len(self.image_ids)


    def __getitem__(self, image_index):
        '''
        Get the specified item from the dataset.

        Params
        :image_index (int) -> id of the image (corresponding to the json file).

        Return <if there is a policy>
        :img (torch.tensor) -> image already normalized.
        :boxes (torch.tensor) -> annotations. Shape -> [batch , bounding_boxes , 5]. 5 -> first 4 for the coordinates and the final position for the category.
        :imgName (string) -> original name of the image.
        :img_aug (torch.tensor) -> augmented image (resulting image from the library bbaug).
        :bbs_aug (torch.tensor) -> annotations. Shape -> [batch , bounding_boxes , 5]. 5 -> first 4 for the coordinates and the final position for the category.
        :imgName_aug (string) -> name of the augmented image (same name of the original but with 'aug_' added at the beginning).

        Return <if there is NO policy>
        :img (torch.tensor) -> image already normalized.
        :boxes (torch.tensor) -> annotations. Shape -> [batch , bounding_boxes , 5]. 5 -> first 4 for the coordinates and the final position for the category.
        :imgName (string) -> original name of the image.
        '''

        # get the name and read the image
        imgName = self.coco.loadImgs(self.image_ids[image_index])[0]['file_name']
        img = cv2.imread(os.path.join(self.root_dir, self.set_name, imgName))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # List: get annotation id from coco
        ann_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)

        # Dictionary: target coco_annotation file for an image
        coco_annotation = self.coco.loadAnns(ann_ids)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        labels = []
        for i in range(len(coco_annotation)):
            label = coco_annotation[i]['category_id'] - 1
            xmin = float(coco_annotation[i]['bbox'][0])
            ymin = float(coco_annotation[i]['bbox'][1])
            xmax = xmin + float(coco_annotation[i]['bbox'][2])
            ymax = ymin + float(coco_annotation[i]['bbox'][3])
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(int(label))
        
        # Apply augmentation
        if self.policy_container:
            # Select a random sub-policy from the policy list
            random_policy = self.policy_container.select_random_policy()

            # Apply this augmentation to the image, returns the augmented image and bounding boxes
            # The boxes must be at a pixel level. e.g. x_min, y_min, x_max, y_max with pixel values
            img_aug, bbs_aug = self.policy_container.apply_augmentation(
                random_policy,
                img,
                boxes,
                labels,
            )
            img = img.astype(np.float32) / 255.
            img_aug = img_aug.astype(np.float32) / 255.

            # Add the labels to the boxes
            labels = np.array(labels)
            boxes = np.hstack(( np.array(boxes), np.vstack(labels) )) 
            #bbs_aug = np.array(bbs_aug)

            # normalization
            #--------------
            # create dictionary to apply the transformation 
            sample_original = {'img': img, 'annot': boxes}
            if self.transform:
                sample_original = self.transform(sample_original)
            # get the data from the resulting dictionary
            img_t, boxes_t = sample_original['img'], sample_original['annot']
            #--------------

            # sometimes the augmentation gets rid of the bounding boxes
            if len(bbs_aug.shape) > 1:

                # normalization
                #--------------
                # Format correction from the augmentation: send category to the end of the row 
                bbs_aug = bbs_aug[:, [1,2,3,4,0]].astype(np.float32) 
                # create dictionary to apply the transformation
                sample_augmented = {'img': img_aug, 'annot': bbs_aug}
                if self.transform:
                    sample_augmented = self.transform(sample_augmented)
                # get the data from the resulting dictionary
                img_aug_t, bbs_aug_t = sample_augmented['img'], sample_augmented['annot']
                #--------------

                if self.use_only_aug:
                    return img_aug_t, bbs_aug_t.squeeze(), "aug_" + imgName, torch.tensor([]), torch.tensor([]), ""
                else:
                    return img_t, boxes_t.squeeze(), imgName, img_aug_t, bbs_aug_t.squeeze(), "aug_" + imgName

            # augmentation got rid of bboxes... so, return just the original image
            else:
                if self.use_only_aug:
                    img_aug_t, torch.tensor([]), "aug_" + imgName, torch.tensor([]), torch.tensor([]), ""
                else:
                    return img_t, boxes_t.squeeze(), imgName, torch.tensor([]), torch.tensor([]), ""
                

        # No augmentation at all
        else:
            img = img.astype(np.float32) / 255.

            # locate the category at the end
            boxes = np.hstack(( np.array(boxes), np.vstack(labels) ))

            #create a temporal dictionary to apply the transformations
            sample = {'img': img, 'annot': boxes}
            if self.transform:
                sample = self.transform(sample)

            # recover the data and return
            img_, boxes_ = sample['img'], sample['annot']
            return img_, boxes_.squeeze(), imgName
        

    def collater(self, batch):
        '''
        Collater function. In case that an augmentation was applied, we have to concatenate everything in tensors.

        Params
        :batch (tuple) -> tuple with all the data in the form of tensors.

        Return
        :dictionary -> {images: torch.tensor, annotations: torch.tensor, image_names: list(string)}
        '''
        if self.policy_container:
            # get data from the batch
            imgs, annots, imgs_names, imgs_aug, annots_aug, imgs_names_aug = list(zip(*batch))

            # create list from the unaugmented data
            imgs = [i for i in imgs]
            annots = [i for i in annots]
            imgs_names = [i for i in imgs_names]
            
            # add augmented data if there are bounding boxes
            for i, box_aug in enumerate(annots_aug):
                if box_aug.numel() > 0:
                    imgs.append(imgs_aug[i])
                    annots.append(box_aug)
                    imgs_names.append(imgs_names_aug[i])

            # convert list to tensor
            imgs = torch.stack(imgs)


            # Add padding to the annotations
            #-------------------------------
            # sometimes when there is only one bounding box, the dimension of the annotation is incorrect. E.g. torch.tensor([5]) instead of torch.tensor([1,5])
            annots = [an if len(list(an.shape)) != 1 else an.unsqueeze(dim=0) for an in annots]

            # ask for the image that has more bounding boxes and add padding to the rest of the bounding boxes
            # Why? because the format in which the loss functions needs the data. So, all annotations will have the same dimension filled with -1s.
            max_num_annots = max(annot.shape[0] for annot in annots) 
            if max_num_annots > 0:
                annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1
                for idx, annot in enumerate(annots):
                    if annot.shape[0] > 0:
                        annot_padded[idx, :annot.shape[0], :] = annot
            else:
                annot_padded = torch.ones((len(annots), 1, 5)) * -1
            #-------------------------------

            # return a dictionary
            return {'img': imgs, 'annot': annot_padded, 'img_names': imgs_names}


        # No augmentation at all
        else:
            # get data from the batch
            imgs, annots, imgs_names = list(zip(*batch))

            # create list from the unaugmented data
            imgs = [i for i in imgs]
            imgs = torch.stack(imgs)
            annots = [i for i in annots]
            imgs_names = [i for i in imgs_names]

            # Add padding to the annotations
            #-------------------------------
            # sometimes when there is only one bounding box, the dimension of the annotation is incorrect. E.g. torch.tensor([5]) instead of torch.tensor([1,5])
            annots = [an if len(list(an.shape)) != 1 else an.unsqueeze(dim=0) for an in annots]
            max_num_annots = max(annot.shape[0] for annot in annots)

            if max_num_annots > 0:
                annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

                for idx, annot in enumerate(annots):
                    if annot.shape[0] > 0:
                        annot_padded[idx, :annot.shape[0], :] = annot
            else:
                annot_padded = torch.ones((len(annots), 1, 5)) * -1
            #-------------------------------

            # return a dictionary
            return {'img': imgs, 'annot': annot_padded, 'img_names': imgs_names}
        


class Resizer(object):
    """Convert ndarrays in sample to Tensors AND apply resizing to the image and annotations."""
    
    def __init__(self, img_size=512):
        '''
        Resizing the image into a fixed value.

        Params
        :img_size (int) -> size of the output image.
        '''
        self.img_size = img_size
        self.to_tensor = transforms.ToTensor()

    def __call__(self, sample):
        '''
        Method that will be executed when the transformation is called.

        Params
        :sample (dict{numpy, numpy}) -> dictionary with the image and the annotations. Format -> {'img': numpy, 'annot': numpy}

        Return
        :dictionary{torch.tensor, torch.tensor}, same format as the input. Format -> {'img': torch.tensor, 'annot': torch.tensor}
        '''
        image, annots = sample['img'], sample['annot']

        # we try to preserve the ration of the image
        height, width, _ = image.shape
        if height > width:
            scale = self.img_size / height
            resized_height = self.img_size
            resized_width = int(width * scale)
        else:
            scale = self.img_size / width
            resized_height = int(height * scale)
            resized_width = self.img_size

        # resize the image using cv2 and an interpolation
        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        # add zero-padding if necessary
        new_image = np.zeros((self.img_size, self.img_size, 3))
        new_image[0:resized_height, 0:resized_width] = image

        # scale the bounding boxes
        annots[:, :4] *= scale

        return {'img': self.to_tensor(new_image).to(torch.float32), 'annot': self.to_tensor(annots)}



class Normalizer(object):
    """Normalization of the data."""

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        '''
        Normalization using fixed values.

        Params
        :mean (float list) -> mean values of the channels.
        :std (float list) -> standard deviation values of the channels.
        '''
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])

    def __call__(self, sample):
        '''
        Apply normalization when the transformation is called.

        Params
        :sample (dict) -> dictionary with the array containing the image and the annotations. Format -> {'img': img, 'annot': boxes}
        '''
        image, annots = sample['img'], sample['annot']
        return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'annot': annots}


