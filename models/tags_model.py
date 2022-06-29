import sys
import os
import os.path as osp
sys.path.append(os.getcwd())
from models.base_model import BaseCVServiceModel

import tensorflow as tf
import numpy as np
import cv2
import six
import skimage
from typing import Any, Union

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
class TagsModel(BaseCVServiceModel):
    def __init__(self, path_prefix):
        physical_devices= tf.config.list_physical_devices('GPU')
        self.path_prefix = path_prefix
        for d in physical_devices:
            tf.config.experimental.set_memory_growth(d, True)
        tf.config.set_visible_devices(physical_devices[0], 'GPU')
        model_path = osp.join(self.path_prefix, "checkpoints/tags/model-resnet_custom_v4.h5")
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.width = self.model.input_shape[2]
        self.height = self.model.input_shape[1]
        self.tags = self.load_tags(osp.join(self.path_prefix, "checkpoints/tags/tags.txt"))
        print('tags model loaded')
    
    def load_tags(self, tags_path):
        with open(tags_path, "r") as tags_stream:
            tags = [tag for tag in (tag.strip() for tag in tags_stream) if tag]
            return tags

    def load_image_for_evaluate(self,
        input_: Union[str, six.BytesIO], width: int, height: int, normalize: bool = True
        ) -> Any:
        if isinstance(input_, six.BytesIO):
            image_raw = input_.getvalue()
        else:
            image_raw = tf.io.read_file(input_)
        image = tf.io.decode_png(image_raw, channels=3)

        image = tf.image.resize(
            image,
            size=(height, width),
            method=tf.image.ResizeMethod.AREA,
            preserve_aspect_ratio=True,
        )
        image = image.numpy()  # EagerTensor to np.array
        image = self.transform_and_pad_image(image, width, height)

        if normalize:
            image = image / 255.0

        return image


    def transform_and_pad_image(
        self,
        image,
        target_width,
        target_height,
        scale=None,
        rotation=None,
        shift=None,
        order=1,
        mode="edge",):
        """
        Transform image and pad by edge pixles.
        """
        image_width = image.shape[1]
        image_height = image.shape[0]
        image_array = image

        # centerize
        t = skimage.transform.AffineTransform(
            translation=(-image_width * 0.5, -image_height * 0.5)
        )

        if scale:
            t += skimage.transform.AffineTransform(scale=(scale, scale))

        if rotation:
            radian = (rotation / 180.0) * math.pi
            t += skimage.transform.AffineTransform(rotation=radian)

        t += skimage.transform.AffineTransform(
            translation=(target_width * 0.5, target_height * 0.5)
        )

        if shift:
            t += skimage.transform.AffineTransform(
                translation=(target_width * shift[0], target_height * shift[1])
            )

        warp_shape = (target_height, target_width)

        image_array = skimage.transform.warp(
            image_array, (t).inverse, output_shape=warp_shape, order=order, mode=mode
        )

        return image_array

    def run(self, path, **kwargs):
        thres = kwargs.get('thres', 0.1)
        image = self.load_image_for_evaluate(path, width=self.width, height=self.height)
        image_shape = image.shape
        image = image.reshape(
            (1, image_shape[0], image_shape[1], image_shape[2]))
        y = self.model.predict(image)[0].tolist()
        result = list()
        for idx, conf in enumerate(y):
            if conf>thres:
                result.append({"tag": self.tags[idx], "score":conf})
        return result


if __name__ == '__main__':
    model = TagsModel(path_prefix=os.getcwd())
    image_path = 'assets/test.jpg'
    res = model.run(image_path)
    print(res)


