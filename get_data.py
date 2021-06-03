import cv2
import time
import os

time_between = 0
mode = "C"  # V:video or C:camera
origin_file = "origin_data"
save_num = 300

video_name = "Av50126219.mp4"
jump_time = 123  # seconds

if mode == "C":
    timestamp_str = str(time.time())
    road_save = origin_file + "\\"+timestamp_str+"_"+str(time_between)
    os.makedirs(road_save)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    for i in range(save_num):
        time.sleep(time_between)
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        cv2.waitKey(1)
        cv2.imwrite(road_save + "\\" + str(i) + ".jpg", frame)
        print(road_save + "\\" + str(i) + ".jpg")
        print(i)
elif mode == "V":
    assert video_name in os.listdir(origin_file), f"Error:No such video named {video_name} in {origin_file}"
    road_save = origin_file + "\\"+video_name[:-4]+"_"+str(time_between)+"_"+str(jump_time)
    os.makedirs(road_save)

    cap = cv2.VideoCapture(origin_file+"\\"+video_name)
    FPS = cap.get(5)
    timerate = time_between
    framerate = int(timerate * FPS)

    for j in range(int(jump_time*FPS)):
        ret, frame = cap.read()
        cv2.waitKey(0)

    for i in range(save_num):
        for temp in range(framerate):
            ret, frame = cap.read()
            cv2.waitKey(0)
        if ret:
            cv2.imwrite(road_save + "\\" + str(i) + ".jpg", frame)
            print(road_save + "\\" + str(i) + ".jpg")
            print(i)

cap.release()
cv2.destroyAllWindows()
