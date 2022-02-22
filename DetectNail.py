

import cv2
from matplotlib.pyplot import contour
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import matplotlib.pyplot as plt

def DetectPositionMaxSkin(filename,x, y, w, h, lower, upper):
    #y=y+50
    Image = cv2.VideoCapture(filename)
    
    #Image = cv2.VideoCapture('t8.mp4')
    success, frame = Image.read()
    
    while success :
        success, frame = Image.read()
        #cv2.imshow('Imagem Original', frame)
        if success:
            cropeedIMAGE = frame[y:y+h, x:x+w]
            
            converted = cv2.cvtColor(cropeedIMAGE, cv2.COLOR_BGR2HSV)
            #cv2.imshow('convertedHSV',converted)
            skinMask = cv2.inRange(converted, lower, upper)
            #cv2.imshow('skin',skinMask)

            # apply a series of erosions and dilations to the mask
            # using an elliptical kernel            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
            skinMask = cv2.erode(skinMask, kernel, iterations=3)
            skinMask = cv2.dilate(skinMask, kernel, iterations=3)

            # blur the mask to help remove noise, then apply the
            # mask to the frame
            skinMask = cv2.GaussianBlur(skinMask, (11, 11), 5)
            #cv2.imshow('skinMask',skinMask)
            skin = cv2.bitwise_and(cropeedIMAGE, cropeedIMAGE, mask=skinMask)
            #cv2.imshow('skin',skin)


            ########################################################
            #lowerFinger =np.array([8, 15, 110], dtype="uint8")
            #upperFinger = np.array([8, 15, 110], dtype="uint8")
                
            hsv_img = cv2.cvtColor(skin, cv2.COLOR_BGR2HSV)
            #hsv_img = cv2.inRange(hsv_img, lowerFinger, upperFinger)
            #cv2.imshow('hsv_img', hsv_img)

            # Extracting Saturation channel on which we will work
            img_s = hsv_img[:, :, 1]
            #img_s = skin[:, :, 1]
            #cv2.imshow('img_s', img_s)

            # smoothing before applying  threshold
            img_s_blur = cv2.GaussianBlur(img_s, (7,7), 2)  
            #img_s_blur = cv2.medianBlur(skin,5)
            #cv2.imshow('img_s_blur', img_s_blur)
            
            img_s_binary = cv2.threshold(img_s_blur, 200, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]  # Thresholding to generate binary image (ROI detection)
            #cv2.imshow('img_s_binary1', img_s_binary1)

            # reduce some noise
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
            img_s_binary = cv2.morphologyEx(img_s_binary, cv2.MORPH_OPEN, kernel, iterations=4) 
            #cv2.imshow('img_s_binary1', img_s_binary)

            # ROI only image extraction & contrast enhancement, you can crop this region 
            #img_croped = cv2.bitwise_and(img_s, img_s_binary) * 10 
            #cv2.imshow('img_croped', img_croped)
            
             #  eliminate
            kernel = np.ones((5, 5), np.float32)/25
            processedImage = cv2.filter2D(img_s_binary, -1, kernel)
            img_s_binary[processedImage > 250] = 0
            #cv2.imshow('img_s_binary2', img_s_binary)


            edges = cv2.threshold(img_s_binary, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            #th3 = cv2.adaptiveThreshold(img_s_blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
            #_,edges = cv2.threshold(img_croped, 160, 255, cv2.THRESH_BINARY_INV)
            
            #cv2.imshow('edges', edges)
            
            
            #https://docs.opencv.org/3.4/da/d0c/tutorial_bounding_rects_circles.html
            contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            #print("Number of contours =" + str(len(contours)))
            #print("Number of hierarchy =" + str(len(hierarchy)))
            #print(np.argmax(hierarchy))

            contours_poly = [None]*len(contours)
            centers = [None]*len(contours)
            radius = [None]*len(contours)
            #area= [None]*len(contours)
            
            #drawing = np.zeros((edges.shape[0], edges.shape[1], 3), dtype=np.uint8)
            for i, c in enumerate(contours):
                contours_poly[i] = cv2.approxPolyDP(c, 0.02, True)
                #boundRect[i] = cv2.boundingRect(contours_poly[i])
                centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])
                #area[i] = cv2.contourArea(contours[i])
                #print("area: %s" % area)
                #if i>=6 and cv2.contourArea(contours[i]) >= 100:
                if  5000 >= cv2.contourArea(contours[i]) <= 7600 and radius[i] < 50:  
                    #cv2.drawContours(skin, contours_poly, i, (255,0,0))
                    cv2.circle(skin, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), (0,0,255), 2)
                    #cv2.imshow('Contours', skin)
                    cv2.imshow('skin', skin)
    
                    #print((centers[i][0]))S
                    #xe=np.arange(1,121)
                    #print(len(xe))
                    #plt.plot(x,centers[i][0],'ro')
                    #plt.ylabel('some numbers')
                    #plt.show()
    
            
                     
        #cv2.imshow('Skin Mask', skinMask)
        #cv2.imshow('Skin', skin)
        
        #vcat = cv2.hconcat((skinMask, skin))
        #cv2.imshow('vcat', vcat)
        
        
        #cv2.imshow('hsv_img', hsv_img)
        #cv2.imshow('Extracting Saturation', img_s)
        #cv2.imshow('img_s_binary1', img_s_binary1)
        #cv2.imshow('img_croped', img_croped)
        #cv2.imshow('edges', edges)

        

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()

    #xc,yc,wc,hc,skin,skinMask,hsv_img,img_s_blur,img_s_binary1,img_croped,edges,cropeedIMAGE