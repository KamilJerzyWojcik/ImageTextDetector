import cv2
import numpy as np


class ImageService:
    def get_words_from_image(self, boxes, image):
        text_images = []
        img = image.copy()
        img_new = image.copy()
        for box in boxes:
            poly = np.array(box).astype(np.int32).reshape((-1))
            poly = poly.reshape(-1, 2)
            rect = cv2.boundingRect(poly)
            x, y, w, h = rect
            mask = np.zeros(img.shape, dtype=np.uint8)
            roi_corners = poly
            channel_count = img_new.shape[2]
            ignore_mask_color = (255,)*channel_count
            cv2.fillConvexPoly(mask, roi_corners, ignore_mask_color)
            img_masked = cv2.bitwise_and(img_new, mask)
            text_images.append(img_masked[y:y+h, x:x+w].copy())
        return text_images