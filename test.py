import pyrealsense2 as rs
import cv2
import numpy as np
import time

pipeline = rs.pipeline()
config = rs.config()
liner = rs.align(rs.stream.color)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

flag = 0
index = 0
count = 0
start = 0
end = 0

try:
    while True:
        start = time.time()
        frames = pipeline.wait_for_frames()
        lined = liner.process(frames)

        depth_frame = lined.get_depth_frame()
        color_frame = lined.get_color_frame()

        if not depth_frame or not color_frame:
            continue
    
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        #depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha = 0.03), cv2.COLORMAP_JET)

        cv2.imshow('Color', color_image)
        cv2.imshow('Depth', depth_image)

        if flag == 0:
            key = cv2.waitKey(100) & 0xFF 
            if (key == ord("w")): #Start Program
                flag = 1
                print("Started!\n-------------")
            elif (key == ord("q")):
                break
        else:
            count += 1
            filename_depth = "/home/dudl3443/Desktop/RealSenseTest/testImages/depthFrames/frame_{}.png".format(index)
            filename_color = "/home/dudl3443/Desktop/RealSenseTest/testImages/colorFrames/frame_{}.png".format(index)
            index += 1
            cv2.imwrite(filename_depth, depth_image)
            cv2.imwrite(filename_color, color_image)
            if cv2.waitKey(18) & 0xFF == ord('s'): #End Program
                print("\n-------------\nStopped!")
                index = 0
                flag = 0
                count = 0
            end = time.time()
            '''
            if (end-start < 0.0333):
                time.sleep(0.0333-(end-start))
            end = time.time()
            '''
            if (count != 0 and count % 10 == 0):
                print(f"\r{1/(end-start)}", end="")
        

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    