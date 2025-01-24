# Run this to start the live webcam footage.
# This script may take some time to run (30 seconds - 1 minute) because of the cv2.VideoCapture function. (fixed)
# If there is previous data from a prior run in the current working directory, make sure to move it to another folder.
# This includes '.avi' and '.csv' files.

# MAKE THIS ONE ABOUT MULTIPLE REGIONS
import os
import time

import cv2
import numpy as np
import pandas as pd
import datetime
import av
import io
import matplotlib.pyplot as plt
import utils
from utils import file_check

# https://stackoverflow.com/questions/73609006/how-to-create-a-video-out-of-frames-without-saving-it-to-disk-using-python
# Video capturing code taken from here

# The script in CheckCameras.py finds what possible numbers to input to cv2.VideoCapture.
# Adjust 'camera' if the script is outputting the wrong camera.
# This is very finicky since openCV doesn't give information about what number correlates to what camera.
# There will be a lot of trial and error figuring out the right camera, because the number can hop around.
camera = 0
warning_sign_length = 60

print("Hold down your mouse and move it to select the region of interest")
print("Press 'q' once finished to move on. Make sure NUMLOCK is locking the number pad.")

vid = cv2.VideoCapture(camera, cv2.CAP_DSHOW)
fps = int(vid.get(cv2.CAP_PROP_FPS))
width  = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
color = (0, 0, 0)
thickness = 3

starts = []
ends = []
start = None
end = None
drawing = False

# This detects mouse movements/inputs for the region of interest (ROI).
def draw_rectangle(event, x, y, flags, param):
    global start, end, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        start = (x, y)
        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            end = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end = (x, y)


cv2.namedWindow("rectangle")
cv2.setMouseCallback("rectangle", draw_rectangle)


# This is the loop where the ROI is drawn.
print("Press 'u' if you want to save a region of interest")
while True:
    _, image = vid.read()
    if starts and ends:
        for i in range(len(starts)):
            image = cv2.rectangle(image, starts[i], ends[i], color, thickness)

    if start and end:
        image = cv2.rectangle(image, start, end, color, thickness)

    cv2.imshow("rectangle", image)

    # Press the video window and then 'q' to quit and move on.
    # MAKE SURE NUMLOCK IS TURNED ON (can't press the number keys)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    if key == ord('u'):
        starts.append(start)
        ends.append(end)

vid.release()
cv2.destroyAllWindows()

if os.path.exists('output.mp4'):
    if not input("Type yes if you have moved 'output.mp4': ") in utils.possible_inputs:
        print("'output.mp4' will be replaced.")

print("If you don't want to detect a certain value, then type '256' because that is higher than the max difference.")

# Once the ROI is set, users are asked to input the RGB color changes to detect.
red_ui = input("Enter the RED value change to detect as an integer: ")
green_ui = input("Enter the GREEN value change to detect as an integer: ")
blue_ui = input("Enter the BLUE value change to detect as an integer: ")

user_inputs = [red_ui, green_ui, blue_ui]

print("Once done taking measurements, press 'q' to save and export the data.")

vid = cv2.VideoCapture(camera, cv2.CAP_DSHOW)

colors = []
color_change_data = []
colors_per_second = []
notes = []

frame_counter = 0
warning_counter = 0
prev_color = None
warning = False
font = cv2.FONT_HERSHEY_SIMPLEX

output_memory_file = io.BytesIO()
output = av.open(output_memory_file, 'w', format="mp4")
stream = output.add_stream('h264', fps)
stream.width = width
stream.height = height
stream.pix_fmt = 'yuv420p'
stream.options = {'crf': '17'}  # Lower crf = better quality & more file space.

# fig, ax = plt.subplots(1, 3, figsize=(6, 2))
# x_data = []
# red_plot = []
# green_plot = []
# blue_plot = []
# x_axis_counter = 0

start_time = time.time()
num_regions = len(starts)

while True:
    _, frame = vid.read()
    end_time = time.time()
    time_diff = end_time - start_time

    # This section detects change in color based on user input and displays a warning sign.
    if frame_counter == 1:
        prev_color = np.array(colors[-1])
    # The warning sign will be on for 90 frames (3 seconds).
    if warning_counter == warning_sign_length:
        warning = False
    # Checks for color change every 31st frame (approximately every second) and then resets.
    if time_diff >= 1:
        start_time = end_time
        frame_counter = 0
        test_color = np.array(colors[-1])
        #color_diff = [abs(x - y) for x, y in zip(test_color, prev_color)] #[B, G, R]
        color_diff = np.abs(test_color - prev_color)
        print(color_diff)
        for reg in range(num_regions):
            for i in range(3): # The -1 avoids the current time.
                color_value = color_diff[reg][reg][i]
                if isinstance(color_value, np.float64):
                    if color_value > int(user_inputs[i]):
                        warning = True
                        warning_counter = 0
                        current_time = datetime.datetime.now()
                        # data = (current_time, color_diff[reg, i, 0], color_diff[reg, i, 1], color_diff[reg, i, 2],
                        #                           len(colors) + 1, len(colors_per_second) + 1, i,
                        #         test_color[reg, 0], test_color[reg, 1], test_color[reg, 2])
                        # color_change_data.append(data)
                        break

    # If a color change is detected, a warning message is displayed.
    if warning:
        text = 'COLOR CHANGE DETECTED'
        (text_width, text_height), _ = cv2.getTextSize(text, font, 1, 3)
        x = (width - text_width) // 2
        y = height // 8 + text_height // 2
        frame = cv2.putText(frame, text, (x, y), font, 1,
                            (255, 0, 0), 3)

    for i in range(num_regions):
        frame = cv2.rectangle(frame, starts[i], ends[i], color, thickness)

    # This finds the average of all the pixel values in the square for one frame
    reds = [[]] * num_regions
    greens = [[]] * num_regions
    blues = [[]] * num_regions
    frame_average_color = [[]] * num_regions

    for i in range(len(starts)):
        for r in range(starts[i][1] + thickness, ends[i][1] - thickness):
            for c in range(starts[i][0] + thickness, ends[i][0] - thickness):
                pixel = frame[r][c] # List of the 3 RGB values as [B, G, R]
                blues[i].append(pixel[0])
                greens[i].append(pixel[1])
                reds[i].append(pixel[2])

        average_red = np.mean(reds[i])
        average_green = np.mean(greens[i])
        average_blue = np.mean(blues[i])


        current_time = datetime.datetime.now()
        frame_average_color[i].append([average_red, average_green, average_blue])
        # frame_average_color[i].append([average_red, average_green, average_blue, current_time])
    colors.append(frame_average_color) #change later to account for data file

    frame_counter += 1
    if warning_counter < warning_sign_length:
        warning_counter += 1

    if time_diff >= 1:
        colors_per_second.append(frame_average_color)

    image = av.VideoFrame.from_ndarray(frame, format='bgr24')
    packet = stream.encode(image)
    output.mux(packet)  # Write the encoded frame to MP4 file.

    # # Live data visualization
    # x_data.append(x_axis_counter)
    # x_axis_counter += 1
    # red_plot.append(frame_average_color[0])
    # green_plot.append(frame_average_color[1])
    # blue_plot.append(frame_average_color[2])
    #
    # ax[0].clear()
    # ax[1].clear()
    # ax[2].clear()
    #
    # ax[0].plot(x_data, red_plot, color='r', label='Red')
    # ax[1].plot(x_data, green_plot, color='g', label='Green')
    # ax[2].plot(x_data, blue_plot, color='b', label='Blue')
    #
    # ax[0].legend()
    # ax[1].legend()
    # ax[2].legend()
    #
    # plt.draw()
    # plt.pause(0.01)

    cv2.imshow("Live webcam video", frame)
    # plt.show(block=False)


    # Press the video window and then 'q' to quit and export the color data
    # MAKE SURE NUMLOCK IS TURNED ON (can't press the number keys)
    key = cv2.waitKey(1)
    if key == ord('u'):
        current_time = datetime.datetime.now()
        notes.append([len(colors) + 1, len(colors_per_second), current_time])
    if key & 0xFF == ord('q'):
        break

vid.release()
packet = stream.encode(None)
output.mux(packet)
output.close()
cv2.destroyAllWindows()
# plt.close()

with open("output.mp4", "wb") as f:
    f.write(output_memory_file.getbuffer())


color_df = pd.DataFrame(colors, columns=['Red', 'Green', 'Blue', 'Current time: Date / HH:MM:SS'])
colors_per_second_df = pd.DataFrame(colors_per_second, columns=['Red', 'Green', 'Blue',
                                                                'Current time: Date / HH:MM:SS'])
color_change_df = pd.DataFrame(color_change_data, columns=['Current time: Date / HH:MM:SS',
                                                           'Red Difference',
                                                           'Green Difference',
                                                           'Blue Difference',
                                                           'Color Table Row Number',
                                                           'Colors per Second Table Row Number',
                                                           'Color Detected (red=0, green=1, blue=2)',
                                                           'Red Value',
                                                           'Blue Value',
                                                           'Green Value'])
notes_df = pd.DataFrame(notes, columns=['Color Table Row Number of note',
                                        'Colors per Second Table Row  Number of note',
                                        'Current time: Date / HH:MM:SS'])






# This webcam is 30 FPS, which means that each second gives 30 rows of color data
# Can rename 'colorData'. This is the table of all the RGB values throughout the reaction.
file_check('colorData.csv', color_df, 'colorData.csv')

# Can rename 'colorsPerSecondData'. This is the table of the RGB values at each second to shorten Excel.
file_check('colorsPerSecondData.csv', colors_per_second_df, 'colorsPerSecondData.csv')

# Can rename 'colorChangeData'. This is the data for when a color change is detected.
file_check('colorChangeData.csv', color_change_df, 'colorChangeData.csv')

file_check('notes.csv', notes_df, 'notes.csv')

# All of these files are saved into the current working directory (CWD).
# Make sure to transfer the data files somewhere else if it needs to be referenced later.















# 1/15
# Fix fps

# 1/14
# add failsafe for video saving
# add failsafe for ROI and RGB change detection overlap
# Eventually add type in coordinates feature? Note that the top left is (0,0)
# Add way to live show data
# Add way to add notes about what is happening







