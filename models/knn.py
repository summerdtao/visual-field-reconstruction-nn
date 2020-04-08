import os

import cv2
import numpy as np


class KNN:
    def __init__(self, img_path, save_path):
        self.bitmapping = cv2.imread(img_path, 0)
        self.save_path = save_path
        self.width = self.bitmapping.shape[1]
        self.height = self.bitmapping.shape[0]

    def __call__(self):
        pixels = self._do_knn()
        self._save_image(pixels)
        return pixels

    def _save_image(self, data):
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        cv2.imwrite(f"{self.save_path}image_knn.png", data)

    def _do_knn(self):
        base_image = np.zeros((self.height, self.width))

        for h in range(self.height):
            for w in range(self.width):
                if h == 0:
                    h_pixel_set = [0, h + 1]
                elif h == self.height - 1:
                    h_pixel_set = [h - 1, h]
                else:
                    h_pixel_set = [h - 1, h, h + 1]

                if w == 0:
                    w_pixel_set = [0, w + 1]
                elif w == self.width - 1:
                    w_pixel_set = [w - 1, w]
                else:
                    w_pixel_set = [w - 1, w, w + 1]

                for i in h_pixel_set:
                    for j in w_pixel_set:
                        base_image[h, w] += self.bitmapping[i, j]

                base_image[h, w] = base_image[h, w] // (len(w_pixel_set) + len(h_pixel_set))
        return base_image


if __name__ == '__main__':
    bitmap_path = f"./ml/processed.png"
    path_to_save = f"./ml/"
    test_knn = KNN(bitmap_path, path_to_save)
    print(test_knn())
