import itertools
import torch
import torch.nn as nn
import numpy as np
import cv2

##----------------------------------------------------------------

def save(path, image, jpg_quality=None, png_compression=None):
  '''
  persist :image: object to disk. if path is given, load() first.
  jpg_quality: for jpeg only. 0 - 100 (higher means better). Default is 95.
  png_compression: For png only. 0 - 9 (higher means a smaller size and longer compression time).
                  Default is 3.
  '''
  if jpg_quality:
    cv2.imwrite(path, image, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
  elif png_compression:
    cv2.imwrite(path, image, [int(cv2.IMWRITE_PNG_COMPRESSION), png_compression])
  else:
    cv2.imwrite(path, image)

##-----------------------------------------------------------------
class BBoxTransform(nn.Module):
    def forward(self, anchors, regression):
        """
        decode_box_outputs adapted from 
        https://github.com/google/automl/blob/6f6694cec1a48cdb33d5d1551a2d5db8ad227798/efficientdet/anchors.py
            
        Transforms relative regression coordinates to absolute positions.
        Network predictions are normalized and relative to a given anchor; this
        reverses the transformation and outputs absolute coordinates for the input
        image.
          
        Args:
            anchors: anchors on all feature levels.
                     [batchsize, boxes, (y1, x1, y2, x2)]
                     
            regression: box regression targets.
                        [batchsize, boxes, (dy, dx, dh, dw)]
        
        Returns:
            outputs: bounding boxes.

        """
        y_centers_a = (anchors[..., 0] + anchors[..., 2]) / 2
        x_centers_a = (anchors[..., 1] + anchors[..., 3]) / 2
        ha = anchors[..., 2] - anchors[..., 0]
        wa = anchors[..., 3] - anchors[..., 1]

        w = regression[..., 3].exp() * wa
        h = regression[..., 2].exp() * ha

        y_centers = regression[..., 0] * ha + y_centers_a
        x_centers = regression[..., 1] * wa + x_centers_a

        ymin = y_centers - h / 2.
        xmin = x_centers - w / 2.
        ymax = y_centers + h / 2.
        xmax = x_centers + w / 2.

        return torch.stack([xmin, ymin, xmax, ymax], dim=2)


class ClipBoxes(nn.Module):

    def __init__(self):
        super(ClipBoxes, self).__init__()

    def forward(self, boxes, img):
        batch_size, num_channels, height, width = img.shape

        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
        boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

        boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width - 1)
        boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height - 1)

        return boxes


class Anchors(nn.Module):
    """
    adapted and modified from https://github.com/google/automl/blob/master/efficientdet/anchors.py by Zylo117
    """

    def __init__(self, anchor_scale=4., pyramid_levels=None, **kwargs):
        super().__init__()
        self.anchor_scale = anchor_scale

        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]
        else:
            self.pyramid_levels = pyramid_levels

        self.strides = kwargs.get('strides', [2 ** x for x in self.pyramid_levels])
        self.scales = np.array(kwargs.get('scales', [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]))
        self.ratios = kwargs.get('ratios', [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)])

        self.last_anchors = {}
        self.last_shape = None

    def forward(self, image, dtype=torch.float32):
        #print(self.pyramid_levels)
        #print(self.strides)
        #print(self.anchor_scale)
        """Generates multiscale anchor boxes.

        Args:
          image_size: integer number of input image size. The input image has the
            same dimension for width and height. The image_size should be divided by
            the largest feature stride 2^max_level.
          anchor_scale: float number representing the scale of size of the base
            anchor to the feature stride 2^level.
          anchor_configs: a dictionary with keys as the levels of anchors and
            values as a list of anchor configuration.

        Returns:
          anchor_boxes: a numpy array with shape [N, 4], which stacks anchors on all
            feature levels.
        Raises:
          ValueError: input size must be the multiple of largest feature stride.
        """
        image_shape = image.shape[2:]
        #print(image_shape)

        if image_shape == self.last_shape and image.device in self.last_anchors:
            return self.last_anchors[image.device]

        if self.last_shape is None or self.last_shape != image_shape:
            self.last_shape = image_shape

        if dtype == torch.float16:
            dtype = np.float16
        else:
            dtype = np.float32
        #--------------------------
        singleLocationBoxes = []
        #--------------------------
        boxes_all = []
        for stride in self.strides:
            boxes_level = []
            for scale, ratio in itertools.product(self.scales, self.ratios):
                if image_shape[1] % stride != 0:
                    raise ValueError('input size must be divided by the stride.')
                base_anchor_size = self.anchor_scale * stride * scale
                anchor_size_x_2 = base_anchor_size * ratio[0] / 2.0
                anchor_size_y_2 = base_anchor_size * ratio[1] / 2.0

                x = np.arange(stride / 2, image_shape[1], stride)
                y = np.arange(stride / 2, image_shape[0], stride)
                xv, yv = np.meshgrid(x, y) #Return coordinate matrices from coordinate vectors.
                xv = xv.reshape(-1)
                yv = yv.reshape(-1)
                #print(xv)
                #print(yv)
                ## at the three different scales and ratios will have the same coordinates per stride
                # y1,x1,y2,x2
                boxes = np.vstack((yv - anchor_size_y_2, xv - anchor_size_x_2,
                                   yv + anchor_size_y_2, xv + anchor_size_x_2))
                
                
                
                boxes = np.swapaxes(boxes, 0, 1)
                #--------------------------
                singleLocationBoxes.append(np.expand_dims(boxes, axis=1)[0][0])
                #--------------------------
                boxes_level.append(np.expand_dims(boxes, axis=1))
            # concat anchors on the same level to the reshape NxAx4
            boxes_level = np.concatenate(boxes_level, axis=1)
            boxes_all.append(boxes_level.reshape([-1, 4]))
        
        
        anchor_boxes = np.vstack(boxes_all)

        anchor_boxes = torch.from_numpy(anchor_boxes.astype(dtype)).to(image.device)
        anchor_boxes = anchor_boxes.unsqueeze(0)
        
        #--------------------------    
        
        
        if(False):
            print(len(singleLocationBoxes))
            print(len(singleLocationBoxes[0]))
            print(singleLocationBoxes[0])
            import matplotlib.pyplot as plt
            
            
            # Initialize black image of same dimensions for drawing the rectangles
            #blk = np.zeros(white_image.shape, np.uint8)
            init_idx = 0
            #pyramid_colors = [(0,0,255),(255,0,127),(0,204,0),(255,128,0),(0,0,0)]
            #pyramid_colors = [(0,0,255),(255,0,127),(0,153,0),(255,128,0),(0,0,0)]
            pyramid_colors = [(0,0,255),(255,0,127),(0,102,0),(255,128,0),(0,0,0)]
            size_x = 4056
            size_y = 2280
            information_file = open('D:/Manfred/InvestigacionPinas/pineapple-efficientDet/pineapple_efficientdet/test/anchors/anchorSizes.txt', "w")
            
            for pyramid_level, pyramid_color in zip(self.pyramid_levels,pyramid_colors):
                
                information_file.write(f'Pyramid Level: {pyramid_level}' + "\n")
                anchor_num = 1
                for y1,x1,y2,x2 in singleLocationBoxes[init_idx:init_idx+9]:
                    w = abs(x2-x1)
                    h = abs(y2-y1)
                    
                    x1T = int((size_x/2)-w/2)
                    y1T = int((size_y/2)-h/2)
                    x2T = int((size_x/2)+w/2)
                    y2T = int((size_y/2)+h/2)
                    
                    white_image = np.zeros((size_y,size_x,3), np.uint8)
                    white_image.fill(255)
                    cv2.rectangle(white_image, (x1T,y1T), (x2T,y2T), pyramid_color, 2)
                    anchor_w = abs(x1T - x2T)
                    anchor_h = abs(y1T - y2T)
                    
                    outpath_png = f"D:/Manfred/InvestigacionPinas/pineapple-efficientDet/pineapple_efficientdet/test/anchors/p{pyramid_level}_{anchor_num}_{anchor_w}_{anchor_h}.png"
                    information_file.write(f'Anchor number: {anchor_num}, width: {anchor_w}, height:{anchor_h}' + "\n")
                    
                    anchor_num = anchor_num + 1
                    save(outpath_png, white_image,png_compression=4)
                #out = cv2.addWeighted(white_image, 1.0, blk, 0.85, 0)
                #cv2.imwrite(f"D:/Manfred/InvestigacionPinas/pineapple-efficientDet/pineapple_efficientdet/test/anchors/p{pyramid_level}.png", white_image)
            
                
                init_idx = init_idx + 9
        #--------------------------
        # save it for later use to reduce overhead
        #information_file.close()
        self.last_anchors[image.device] = anchor_boxes
        return anchor_boxes
