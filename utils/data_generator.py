import numpy as np
import keras
from skimage.transform import rescale, resize
from keras.preprocessing.image import Iterator, ImageDataGenerator
from io import BytesIO
from PIL import Image    




class DataGenerator(Iterator):
    # Generates data for Keras
    def __init__(self, X, input_dim, batch_size, colormap, datagen_args,
                 y=None, shuffle=True, seed=None):
        
        # Initialization
        self.X = X
        self.y = y
        self.input_dim = np.asarray(input_dim)
        self.scaled_dim = *(self.input_dim[:-1] // 4), self.input_dim[-1]
        self.batch_size = batch_size
        self.colormap = colormap
        self.num_classes = len(colormap)

        if seed is None:
            seed = np.random.randint(0, 1000)
        
        # Some args need to be seperate since they change the masks' colors and
        # could interrupt color mappings. They also must be seperate for input 
        # images since they the change order of np.RandomState
        exclude_list = ['channel_shift_range', 'brightness_range',
                        'zca_whitening', 'zca_epsilon', 'rescale']
        args1 = {k:v for k,v in datagen_args.items() if k not in exclude_list}
        args2 = {k:v for k,v in datagen_args.items() if k in exclude_list}
        self.datagen_args = [args1, args2]

        super(DataGenerator, self).__init__(len(X), batch_size, shuffle, seed)
 

    def next(self):
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch.
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)


    def _get_batches_of_transformed_samples(self, index_array):
        # Generates data containing batch_size samples
        index_array = sorted(index_array)
        imgs = self._get_augmented_images(self.X[index_array], self.datagen_args)
        
        imgs_scaled = np.empty((self.batch_size, *self.scaled_dim))
        for i,img in enumerate(imgs):
            imgs_scaled[i] = rescale(img, 1/4, mode='reflect', multichannel=True,
                                     anti_aliasing=True)         
        if self.y is None:
            return imgs, imgs_scaled
        
        masks = self._get_augmented_images(self.y[index_array], [self.datagen_args[0]])
        one_hots = np.empty((self.batch_size, *self.input_dim[:-1], self.num_classes))
        for i,img in enumerate(masks):
            for c,ldef in enumerate(self.colormap):
                one_hots[i,:,:,c] = np.all(img == np.array(ldef.color), axis=2)
        return (imgs, imgs_scaled), one_hots


    def _get_augmented_images(self, binary_images, datagen_args):
        seed=self.seed + self.total_batches_seen
        np.random.seed(seed)

        imgs = np.empty((self.batch_size, *self.input_dim))
        for i,v in enumerate(binary_images):
            imgs[i] = self.load_img(BytesIO(v), self.input_dim)
        
        for args in datagen_args:
            datagen = ImageDataGenerator(**args)
            datagen.fit(imgs, seed=seed)
            imgs[:] = datagen.flow(imgs, batch_size=self.batch_size,
                                   seed=seed, shuffle=False)[0]
        return imgs
    

    @staticmethod
    def load_img(fp_img, target_size, resample=Image.NEAREST):
        img = Image.open(fp_img) 
        if img.mode != 'RGB':
            img = img.convert('RGB')
    
        width_height_tuple = (target_size[1], target_size[0])
        w, h = img.size
        if target_size[0] <= h and target_size[1] <= w:
            img = DataGenerator.random_crop(img, width_height_tuple)
        else:
            min_sz = min(w, h)
            img = DataGenerator.random_crop(img, (min_sz, min_sz))
            img = img.resize(width_height_tuple, resample)
    
        img = np.asarray(img, dtype=keras.backend.floatx())
        return img
    

    @staticmethod
    def random_crop(img, random_crop_size):
        width, height = img.size # PIL format
        dx, dy = random_crop_size
        x = np.random.randint(0, width - dx + 1)
        y = np.random.randint(0, height - dy + 1)
        return img.crop((x, y, x+dx, y+dy))