import numpy as np
import keras
from skimage.transform import rescale, resize
from keras.preprocessing.image import Iterator, ImageDataGenerator
from io import BytesIO
from PIL import Image    





class zipped_generators(keras.utils.Sequence):
    def __init__(self, x_gen, y_gen):
        self.x_gen = x_gen
        self.y_gen = y_gen
        
    def __getitem__(self, index):
        return (self.x_gen[index], self.y_gen[index])

    def __len__(self):
        return len(self.x_gen)


def build_data_generators(X, y=None, **kwargs):
    image_generator = DataGenerator(X, **kwargs)
    if y is None:
        return image_generator
    
    mask_generator = DataGenerator(y, isImage=False, **kwargs)
    generator = zipped_generators(image_generator, mask_generator)
    return generator


class DataGenerator(Iterator):
    # Generates data for Keras
    def __init__(self, data, input_dim, batch_size, colormap, datagen_args,
                 isImage=True, shuffle=True, seed=None):
        
        # Initialization
        self.data = data
        self.input_dim = np.asarray(input_dim)
        self.scaled_dim = *(self.input_dim[:-1] // 4), self.input_dim[-1]
        self.batch_size = batch_size
        self.colormap = colormap
        self.num_classes = len(colormap)
        self.isImage = isImage

        if seed is None:
            seed = np.random.randint(0, 1000)
        
        self.datagens = []
        ex_list = ['channel_shift_range', 'brightness_range',
                    'zca_whitening', 'zca_epsilon', 'rescale']         
        args1 = {k:v for k,v in datagen_args.items() if k not in ex_list}
        args2 = {k:v for k,v in datagen_args.items() if k in ex_list}
        if len(args1):
            self.datagens.append(ImageDataGenerator(**args1))
        if len(args2) and isImage:
            self.datagens.append(ImageDataGenerator(**args2))

        super(DataGenerator, self).__init__(len(data), batch_size, shuffle, seed)
    

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
        imgs = np.empty((self.batch_size, *self.input_dim))
        for i,idx in enumerate(index_array):
            imgs[i] = self.load_img(BytesIO(self.data[idx]), self.input_dim)
        
        for datagen in self.datagens:
            datagen.fit(imgs, seed=self.seed)
            imgs = datagen.flow(imgs, batch_size=self.batch_size,
                                seed=self.seed, shuffle=False)[0]
        
        
        if self.isImage:
            imgs_scaled = np.empty((self.batch_size, *self.scaled_dim))
            for i,img in enumerate(imgs):
                imgs_scaled[i] = rescale(img, 1/4, mode='reflect', multichannel=True,
                                         anti_aliasing=True)  
            return [imgs, imgs_scaled]

        else:
            labels = np.empty((self.batch_size, *self.input_dim[:-1], self.num_classes))
            for i,img in enumerate(imgs):
                for c,ldef in enumerate(self.colormap):
                    labels[i,:,:,c] = np.all(img == np.array(ldef.color), axis=2)
            return labels
    

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