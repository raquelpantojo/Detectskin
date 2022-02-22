# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 14:53:36 2022

@author: RaquelPantojo
"""

# import the necessary packages
import cv2
from matplotlib.pyplot import contour
import numpy as np
import matplotlib.pyplot as plt


def DetectPositionMaxSkin(filename, frame_number, lower, upper):

    Image = cv2.VideoCapture(filename)
    # keep looping over the frames in the video
    # Get the total number of frames in the video.
    fps = Image.get(cv2.CAP_PROP_FPS)
    frame_count = Image.get(cv2.CAP_PROP_FRAME_COUNT)
    success, frame = Image.read()

    while success and frame_number <= frame_count:

        if success:
            frame_number += fps
            Image.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

            # resize the frame, convert it to the HSV color space,
            # and determine the HSV pixel intensities that fall into
            # the speicifed upper and lower boundaries
            # frame = imutils.resize(frame, width=400)  # 400
            #frame=cv2.resize(frame, (640, 480))

            converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            skinMask = cv2.inRange(converted, lower, upper)

            # apply a series of erosions and dilations to the mask
            # using an elliptical kernel
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
            skinMask = cv2.erode(skinMask, kernel, iterations=3)
            skinMask = cv2.dilate(skinMask, kernel, iterations=3)

            # blur the mask to help remove noise, then apply the
            # mask to the frame
            skinMask = cv2.GaussianBlur(skinMask, (11, 11), 5)
            skin = cv2.bitwise_and(frame, frame, mask=skinMask)
            #skin[ skinMask == 0] = 255
            cv2.imshow('Image skin', skin)

           ###################################################################
           # Parte para o corte da região da pele
            _, thresh = cv2.threshold(skinMask, 40, 255, 0)
            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            for contour in contours:
                # draw in blue the contours that were founded
                cv2.drawContours(skin, contours, -1, (0, 255, 0), 1)

                # find the biggest countour (c) by the area
                c = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(c)
                #print(x, y, w, h)
            ###################################################################

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()

    return x, y, w, h

#}}}


def croppedSkin(filename, x, y, w, h):

    Image = cv2.VideoCapture(filename)
    #Image = cv2.VideoCapture('t8.mp4')
    success, frame = Image.read()

    while success:
        success, frame = Image.read()

        if success:
            cropeedIMAGE = frame[y:y+h, x:x+w]
            #cv2.imshow('Video', sky)
            cv2.imshow('finger', cropeedIMAGE)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    return cropeedIMAGE
