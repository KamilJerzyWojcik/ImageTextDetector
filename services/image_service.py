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

    def get_phrases_from_image(self, boxes, image):
        text_images = []
        img = image.copy()
        img_new = image.copy()
        phrases = self.get_phrases(boxes)
        for phrase in phrases:
            # poly = np.array(phrase).astype(np.int32).reshape((-1))
            # poly = poly.reshape(-1, 2)
            # rect = cv2.boundingRect(poly)
            x, y, w, h = phrase
            # mask = np.zeros(img.shape, dtype=np.uint8)
            # roi_corners = poly
            # channel_count = img_new.shape[2]
            # ignore_mask_color = (255,)*channel_count
            # cv2.fillConvexPoly(mask, roi_corners, ignore_mask_color)
            # img_masked = cv2.bitwise_and(img_new, mask)
            img_r = img.copy()
            text_images.append(img_r[y:y+h, x:x+w])
        return text_images
    
    def get_phrases(self, boxes):
        rects = []

        for box in boxes:
            poly = np.array(box).astype(np.int32).reshape((-1))
            poly = poly.reshape(-1, 2)
            rect = cv2.boundingRect(poly)
            rects.append(rect)
        
        phrases = []
        first_phrase = rects[0]

        phrases.append(first_phrase)
        rects.remove(first_phrase)
    
        while True:
            print(len(rects))
            added_rects = []

            for i, _ in enumerate(phrases):
                for rect in rects:
                    if self.is_overlaps(phrases[i], rect):
                        added_rects.append(rect)
                        phrases[i] = self.union_rectangle(phrases[i], rect)
            
            if added_rects != []:
                for r in added_rects:
                    rects.remove(r)
                added_rects = []
            elif len(rects) > 0:
                next_phrase = rects[0]
                phrases.append(next_phrase)
                rects.remove(next_phrase)
            
            if len(rects) == 0:
                break
        
        return phrases

    def is_overlaps(self, rect1, rect2):
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2

        left = max(x1, x2)
        right = min(x1 + w1, x2 + w2)
        
        top = max(y1, y2)
        bottom = min(y1 + h1, y2 + h2)

        if left <= right and bottom >= top:
            return True
        else:
            return False
    

    def union_rectangle(self, rect1, rect2):
        x = min(rect1[0], rect2[0])
        y = min(rect1[1], rect2[1])
        w = max(rect1[0]+rect1[2], rect2[0]+rect2[2]) - x
        h = max(rect1[1]+rect1[3], rect2[1]+rect2[3]) - y
        return (x, y, w, h)

    def combineRect(self, rectA, rectB): # create bounding box for rect A & B
        a, b = rectA, rectB
        startX = min( a[0], b[0] )
        startY = min( a[1], b[1] )
        endX = max( a[2], b[2] )
        endY = max( a[3], b[3] )
        return (startX, startY, endX, endY)
