# Set up of ESP_32 CAM 
Since The ESP_32 CAM doesn't have a USB port, so we use an FTDI programmer to interface it.

The ESP_32 CAM includes the OV2640 Camera Module.


## Specifications of Module

* 2 Megapixel sensor
* Array size UXGA 1622×1200
* Output formats include YUV422, YUV420, RGB565, RGB555 and 8-bit compressed data
* Image transfer rate of 15 to 60 fps


## Connecting the FTDI to ESP Module

1. Connect the GPIO 3(U0R) receive PIN of ESP32 to TX PIN of FTDI and the GPIO 1(U0T) Transmit PIN of ESP32 to RX Pin of FTDI.
2. Connect the 5V PIN of FTDI to the VCC of ESP32. 
3. Connect GND to GND.
4. Connect the GPIO 0 Pin of ESP32 to GND (Needed until the ESP32 is flashed/programmed by Arduino IDE, Remove the connection after it is flashed.)

![ESP32-CAM-pinouts](https://user-images.githubusercontent.com/54215971/155779981-49d29d69-bfb9-4a7b-b99d-a8c5d41baf3b.jpeg)


## Interfacing with Arduino IDE

First we set up the Arduino boards manager from the preference section of Arduino IDE.  
Then we need to add this 'https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json' to "additional Board Manager URL box".

![Arduino Boards Manager](https://user-images.githubusercontent.com/54215971/155780323-27435746-4c37-428b-9b63-1471cdcde62c.jpg)

Select the AI-Thinker module from the board manager from tools.  
Upload the code [ESP_32_CAM.ino + Base64.cpp + Base64.h] (all the 3 files).
Select the camera model as 'CAMERA_MODEL_AI_THINKER'.  
Provide the SSID and Password of the wi-fi network.


## Creating a Google APPS Script

Create a **upload.gs** script file and deploy the project as a 'web-app'.  
Modify the access to 'anyone' so that the ESP32 can access the file.  
Copy the deployment id and put it into the part of the code "String myScript = /macros/s/ **Deployment ID**".  

![deployment id](https://user-images.githubusercontent.com/54215971/155781398-1b85384e-3146-4a6b-949e-b250f3faebe5.jpg)


## Uploading the Code

Load the sketch into the IDE and upload it to the module (make sure that the GPIO 0 is shorted, also we might need to press the Reset button when the dots appear while the code is uploading).  
After the code is uploaded disconnect the GPIO 0 from GND. 

Open the terminal window now and press the Reset button on the module.  
If everything is done correctly according to the process we would get the success message as "connected to script.google.com" and images will start uploading automatically into the google drive folder for every one minute or so depending on the code.   

![terminal](https://user-images.githubusercontent.com/54215971/155786213-f83effbd-80fa-4e3e-a7e9-5544952e7bcd.jpg)
 
The Drive will get filled up with images just like the image below.  

![image](https://user-images.githubusercontent.com/54215971/155787943-4ec631e7-4862-4c8c-8067-2de68e1cc7b0.jpg)
