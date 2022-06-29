import random
import torch


class ImagePool:

    def __init__(self, pool_size: int):
        self.__pool_size = pool_size
        if pool_size > 0:
            self.__num_imgs = 0
            self.__images = []

    def query(self, images):
        if self.__pool_size == 0:
            return images
        return_images = []
        for image in images.data:
            image = torch.unsqueeze(image, 0)
            if self.__num_imgs < self.__pool_size:
                self.__num_imgs = self.__num_imgs + 1
                self.__images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.__pool_size - 1)
                    tmp = self.__images[random_id].clone()
                    self.__images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = torch.cat(return_images, dim=0)
        return return_images
