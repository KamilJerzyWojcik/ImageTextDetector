import cv2
import numpy as np
from copy import deepcopy


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
    

    def get_sorted_phrases_words(self, boxes):
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
            added_rects = []
            for i, _ in enumerate(phrases):
                for rect in rects:
                    if self.is_overlaps(phrases[i], rect) and self.is_in_line(phrases[i], rect):
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
            

        phrases = self.sort_phrases(phrases)

        return phrases
    
    def sort_phrases(self, phrases):

        first_phrase = phrases[0]
        phrases_boxes = [[first_phrase]]
        phrases.remove(first_phrase)

        while True:
            added_phrases = []
            copy_phrases_boxes = deepcopy(phrases_boxes)

            for phrases_box_i, _ in enumerate(phrases_boxes):
                boxes = phrases_boxes[phrases_box_i]
                for phrase_box_i, _ in  enumerate(boxes):
                    for phrase in phrases:
                        if self.is_in_line(phrase, boxes[phrase_box_i]):
                            copy_phrases_boxes[phrases_box_i].append(phrase)
                            added_phrases.append(phrase)

            phrases_boxes = deepcopy(copy_phrases_boxes)

            if added_phrases != []:
                for r in added_phrases:
                    phrases.remove(r)
                added_phrases = []
            elif len(phrases) > 0:
                next_phrase = phrases[0]
                phrases_boxes.append([next_phrase])
                phrases.remove(next_phrase)
            if len(phrases) == 0:
                break
        

        for i, _ in enumerate(phrases_boxes):
            self.bubbleSort(phrases_boxes[i])
        return phrases_boxes

    def is_in_line(self, phrase, rect):
        x1_left, y1_bottom, w1, h1 = phrase
        x2_left, y2_bottom, w2, h2 = rect

        y1_top = y1_bottom + h1
        y2_top = y2_bottom + h2

        y_top_common = 0
        y_bottom_common = 0

        if y1_top <= y2_top:
            y_top_common = y1_top
        else:
            y_top_common = y2_top
        
        if y2_bottom >= y1_bottom:
            y_bottom_common = y2_bottom
        else:
            y_bottom_common = y1_bottom

        h_common = y_top_common - y_bottom_common

        h1_rate_common = h_common / h1
        h2_rate_common = h_common / h2

        max_rate = min(h1_rate_common, h2_rate_common)

        return max_rate > 0.5

    def bubbleSort(self, ar):
        n = len(ar)
        for i in range(n):
            for j in range(0, n-i-1):
                if ar[j][0] > ar[j+1][0] :
                    ar[j], ar[j+1] = ar[j+1], ar[j]

    def get_sorted_phrases_from_image(self, boxes, image):
        text_images = []
        img = image.copy()
        phrases = self.get_sorted_phrases_words(boxes)
        for phrase in phrases:
            for word in phrase:
                x, y, w, h = word
                img_r = img.copy()
                text_images.append(img_r[y:y+h, x:x+w])
        return phrases, text_images


    def get_phrases_from_image(self, boxes, image):
        text_images = []
        img = image.copy()
        phrases = self.get_phrases(boxes)
        for phrase in phrases:
            x, y, w, h = phrase
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
