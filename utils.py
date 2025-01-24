import cv2
import os
import pandas

possible_inputs = ['yes', ' yes', 'yes ', 'YES', ' YES', 'YES ', "'yes'", "'Yes'", 'ye']


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


def file_check(file_path, dataframe, file_name):
    if os.path.exists(file_path):
        first_yes = input(f"Enter 'yes' to continue after you have moved '{file_path}' to another folder. "
                    f"If not moved, the current file will be overwritten. \n"
                    f"If something else is entered, the old file will stay and the current file will be lost: ")
        if first_yes in possible_inputs:
            try:
                dataframe.to_csv(file_name, mode='w', index=False)
            except PermissionError:
                error_input = input("There is a permission error happening. Try closing the excel."
                                    "Type 'yes' once done.")
                if error_input in possible_inputs:
                    dataframe.to_csv(file_name, mode='w', index=False)
        else:
            second_yes = input('Last chance to save the file. Type "yes" to save the file ')
            if second_yes in possible_inputs:
                try:
                    dataframe.to_csv(file_name, mode='w', index=False)
                except PermissionError:
                    error_input = input("There is a permission error happening. Try closing the excel."
                                        "Type 'yes' once done.")
                    if error_input in possible_inputs:
                        dataframe.to_csv(file_name, mode='w', index=False)
    else:
        dataframe.to_csv(file_name, mode='w', index=False)