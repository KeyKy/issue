import os
import numpy as np
from imdb import Imdb
import cv2
from config.config import cfg
import PIL
import logging
import math


class Fddb(Imdb):
    def __init__(self, image_set, year, devkit_path, shuffle=False, is_train=False):
        logging.info("we do not use shuffle in this FDDB_{} dataset whatever shuffle is {}". \
                        format(image_set, shuffle))
        super(Fddb, self).__init__('fddb_' + year + '_' + image_set)
        self.image_set = image_set
        self.year = year

        self.devkit_path = self._get_default_path() if devkit_path is None \
                            else devkit_path

        self.data_path = self.devkit_path
        self.classes = ('face',)
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self.extension = '.jpg'
        self.is_train = is_train

        self.config = {'padding': 56}

        self.image_set_index = self._load_image_set_index()
        self.num_images = len(self.image_set_index)
        if self.is_train:
            self.labels = self._load_image_labels()
    
    @property
    def cache_path(self):
        cache_path = os.path.join(os.path.dirname(__file__), '..', 'cache')
        if not os.path.exists(cache_path):
            os.mkdir(cache_path)
        return cache_path

    def _get_default_path(self):
        return os.path.join(cfg.ROOT_DIR, 'data', 'fddb' + self.year)
    
    def _load_image_set_index(self):
        image_set_file = os.path.join(self.data_path, 'FDDB-folds', 
                            '-'.join(['FDDB', 'fold', self.image_set]) + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exists: {}'.format(image_set_file)
        with open(image_set_file, 'r') as f:
            image_set_index = [x.strip() for x in f.readlines()]
        
        return image_set_index

    def image_path_from_index(self, index):
        assert self.image_set_index is not None, "Dataset not initialized"
        name = self.image_set_index[index]
        image_file = os.path.join(self.data_path, name+self.extension)
        assert os.path.exists(image_file), 'Path does not exist: {}'.format(image_file)
        return image_file

    def label_from_index(self, index):
        assert self.labels is not None, "Labels not processed"
        return self.labels[index, :, :] # num_images, num_boxes, label_x1_y1_x2_y2

    def _load_image_labels(self):
        filename = os.path.join(self.data_path, 'FDDB-folds',
                        '-'.join(['FDDB', 'fold', self.image_set, 'ellipseList']) + '.txt')
        
        reader = open(filename, 'r')
        temp = []; max_objects = 0; count = 0

        while True:
            line = reader.readline()
            if line == '':
                break
            image_set_index = line.strip()
            assert(self.image_set_index[count] == image_set_index), \
                    'image index at {},{} == {}'.format(count, self.image_set_index[count], image_set_index)
            num_boxes = int(reader.readline().strip())
            ellipsis = np.zeros([num_boxes, 5], dtype=np.float32)
            for i in range(num_boxes):
                line = reader.readline().strip().split(' ')
                ellipse = [float(j) for j in line[:5]]
                ellipsis[i, :] = ellipse
            image_path = self.image_path_from_index(count)
            #import ipdb; ipdb.set_trace()

            width, height = PIL.Image.open(image_path).size # PIL size -> [width, height]
            
            label = self._ellipsis_to_bboxes(ellipsis, width, height)
            temp.append(label)
            max_objects = max(max_objects, label.shape[0])
            count += 1

        assert max_objects >0, "No objects found for any of images"
        assert max_objects <= self.config['padding'], "# obj exceed padding"

        self.padding = self.config['padding']
        labels = []
        for label in temp:
            label = np.lib.pad(label, ((0, self.padding-label.shape[0]), (0,0)), \
                                'constant', constant_values=(-1,-1))
            labels.append(label)
        return np.array(labels)

    def _ellipsis_to_bboxes(self, ellipsis, im_width, im_height):
        num_boxes = ellipsis.shape[0]
        boxes = np.zeros([num_boxes, 5])
        for i in range(num_boxes):
            major_axis_radius, minor_axis_radius, angle, center_x, center_y = ellipsis[i, :]
            dy = 2 * major_axis_radius * math.sin(-angle)
            dx = 2 * minor_axis_radius * math.cos( math.pi/2 - (-angle) )
            width = np.abs(dx); height = np.abs(dy);
            x1 = center_x - width/2.0; y1 = center_y - height/2.0
            x2 = width + x1 - 1; y2 = height + y1 - 1
 
            x1 = x1 if x1 >= 0 else 0.0
            y1 = y1 if y1 >= 0 else 0.0
            x2 = x2 if x2 >= 0 else 0.0
            y2 = y2 if y2 >= 0 else 0.0

            x1 = x1 if x1 < im_width else im_width-1
            y1 = y1 if y1 < im_height else im_height-1
            x2 = x2 if x2 < im_width else im_width-1
            y2 = y2 if y2 < im_height else im_height-1
            
            cls = self._class_to_ind['face']

            xmin = float(x1) / float(im_width)
            ymin = float(y1) / float(im_height)
            xmax = float(x2) / float(im_width)
            ymax = float(y2) / float(im_height)

            boxes[i, :] = [cls, xmin, ymin, xmax, ymax]
        return boxes
