import sys
import os
import time,board,busio
import colorsys
import numpy as np
import cv2
import datetime
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
from PIL import Image
import glob

sys.path.insert(0, "./build/lib.linux-armv7l-3.5")

import adafruit_mlx90640

i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)

img = Image.new( 'L', (24,32), "black")
t_end = time.time() + 10 * 90

def irCounter():


    mlx = adafruit_mlx90640.MLX90640(i2c)
    mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_4_HZ #set frame rate of MLX90640

    frame = np.zeros((24*32,))    
    mlx.getFrame(frame)

    # get max and min temps from sensor
    v_min = np.min(frame)
    v_max = np.max(frame)

    # Console output for testing
    textTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') # get timestamp
    print(textTime)

    for x in range(24):
        row = []
        for y in range(32):
            val = frame[32 * (23-x) + y]
            row.append(val)
            img.putpixel((x, y), (int(val)))

    # convert raw temp data to numpy array
    imgIR = np.array(img)

    ## Threshold the -40C to 300 C temps to a more human range
    # Sensor seems to read a bit cold, calibrate in final setting
    rangeMin = 30 # low threshold temp in C
    rangeMax = 40 # high threshold temp in C


    # Apply thresholds based on min and max ranges
    depth_scale_factor = 255.0 / (rangeMax-rangeMin)
    depth_scale_beta_factor = -rangeMin*255.0/(rangeMax-rangeMin)

    depth_uint8 = imgIR*depth_scale_factor+depth_scale_beta_factor
    depth_uint8[depth_uint8>255] = 255
    depth_uint8[depth_uint8<0] = 0
    depth_uint8 = depth_uint8.astype('uint8')

    # increase the 24x32 px image to 240x320px for ease of seeing
    bigIR = cv2.resize(depth_uint8, dsize=(240,320), interpolation=cv2.INTER_CUBIC)

    # Normalize the image
    normIR = cv2.normalize(bigIR, bigIR, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # Apply a color heat map
    colorIR = cv2.applyColorMap(normIR, cv2.COLORMAP_JET)
    imagepath = '/home/pi/Thermcam/Images'
    cv2.imwrite(os.path.join(imagepath , textTime + " frames.png"), colorIR)
    
    cv2.waitKey(1)



try:
    while time.time() < t_end:
        irCounter()




except KeyboardInterrupt:

    width  = 1280
    height = 720
    channel = 3
    fps = 4
    sec = 900
    
    fourcc = cv2.VideoWriter_fourcc(*'MP42')
    video = cv2.VideoWriter('image_to_video.avi', fourcc, 4, (width, height))

    directry = '/home/pi/Thermcam/Images'
    img_name_list = os.listdir(directry)

     
    for frame_count in range(fps*sec):
        img_name = np.random.choice(img_name_list)
        img_path = os.path.join(directry, img_name)
        sig = cv2.imread(img_path)
        sig_resize = cv2.resize(sig, (width, height))
        video.write(sig_resize)
            
    video.release()

    print('interrupted!')
