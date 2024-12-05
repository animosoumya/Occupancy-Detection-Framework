import cv2
 
capture = cv2.VideoCapture('"D:\books\MS Thesis\resized dataset for traffic signs\yolo\2022-08-26 18h 41m 41s.mp4"')
 
count= 0
 
while (True):
    success, frame = capture.read()
 
    if success:
        cv2.imwrite(f'D:\books\MS Thesis\resized dataset for traffic signs\yolo\\frame_{count}.jpg', frame)
 
    else:
        break
 
    count = count+1
 
capture.release()