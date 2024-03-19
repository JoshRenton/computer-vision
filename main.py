import math
import numpy as np
import cv2
import pandas as pd

def round_angle(angle):
    if angle < 0:
        angle += 180
    if 0 <= angle < 22.5 or 157.5 <= angle <= 180:
        return 0
    elif 22.5 <= angle < 67.5:
       return 45
    elif 67.5 <= angle < 112.5:
        return 90
    else:
        return 135
    
def pixel_suppressed(g, r_angle, row, column):
    r_row, r_col, p_row, p_col = 0, 0, 0, 0
    intensity = g[row][column]

    if r_angle == 0:
        r_row = row
        r_col = column + 1
        p_row = row
        p_col = column - 1
    elif r_angle == 135:
        r_row = row - 1
        r_col = column + 1
        p_row = row + 1
        p_col = column - 1
    elif r_angle == 90:
        r_row = row - 1
        r_col = column
        p_row = row + 1
        p_col = column
    else:
        r_row = row - 1
        r_col = column - 1
        p_row = row + 1
        p_col = column + 1

    if 0 <= r_row < g.shape[0] and 0 <= r_col < g.shape[1]:
        r = g[r_row][r_col]
    else:
        r = 0

    if 0 <= p_row < g.shape[0] and 0 <= p_col < g.shape[1]:
        p = g[p_row][p_col]
    else:
        p = 0

    return not ((intensity >= r) and (intensity >= p))

def canny(image, t_low, t_high):
    image_blur = cv2.GaussianBlur(image, (3, 3), 1)
    
    # Apply Scharr filters to image
    gx = cv2.Scharr(image_blur, cv2.CV_16S, 1, 0)
    gy = cv2.Scharr(image_blur, cv2.CV_16S, 0, 1)

    # Calculate gradient magnitude and direction
    g = np.hypot(gx, gy)
    theta = np.arctan2(gy, gx)
    g = (g / g.max()) * 255

    # Convert to degrees
    theta = np.rad2deg(theta)

    a = np.zeros((g.shape[0], g.shape[1], 3))

    for row in range(0, a.shape[0]):
        for column in range(0, a.shape[1]):
            r_angle = round_angle(theta[row][column])
            if r_angle == 0:
                a[row][column] = [255, 0, 0]
            if r_angle == 45:
                a[row][column] = [0, 255, 0]
            if r_angle == 90:
                a[row][column] = [0, 0, 255]
            if r_angle == 135:
                a[row][column] = [255, 255, 255]

    suppress = np.zeros_like(image, dtype=np.float32)

    # Non-maximum suppression
    for row in range(0, g.shape[0]):
        for column in range(0, g.shape[1]):
            angle = theta[row][column]
            r_angle = round_angle(angle)

            if pixel_suppressed(g, r_angle, row, column) == False:
                suppress[row][column] = g[row][column]

    output = np.zeros_like(suppress)

    output[suppress < t_low] = 0
    output[suppress >= t_low] = t_low
    output[suppress >= t_high] = 255

    # Hysteresis Thresholding
    for row in range(0, g.shape[0]):
        for column in range(0, g.shape[1]):
            if output[row][column] == 255:
                new_r = row
                new_c = column
                intensity = output[row][column]
                while(intensity >= t_low):
                    output[new_r][new_c] = 255
                    angle = theta[new_r][new_c]
                    r_angle = round_angle(angle)

                    # Get row and column of next pixel
                    if r_angle == 0:
                        new_c += 1
                    elif r_angle == 135:
                        new_r -= 1
                        new_c += 1
                    elif r_angle == 90:
                        new_r -= 1
                    else:
                        new_r -= 1
                        new_c -= 1

                    next_angle = theta[new_r][new_c]
                    next_angle = round_angle(next_angle)

                    intensity = output[new_r][new_c]

    output[output < 255] = 0

    cv2.imshow("suppressed", suppress)
    cv2.imshow("output", output)
    cv2.imshow("a", a)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return output.astype(np.uint8)

def testCanny(folder_name):
    image =  cv2.imread(folder_name + '/' + 'image9.png', cv2.IMREAD_GRAYSCALE)
    canny(image, 50, 150)

def testTask1(folder_name):
    # Read in data
    task1_data = pd.read_csv(folder_name+"/list.txt")
    task1_data = task1_data.reset_index()

    actual_angles = []
    predicted_angles = []

    # Iterate through all images
    for index, row in task1_data.iterrows():
        image =  cv2.imread(folder_name + '/' + row['FileName'], cv2.IMREAD_GRAYSCALE)
        actual_angles.append(row['AngleInDegrees'])
        edges = canny(image, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 90, None, 0, 0)

        if lines is None:
            print(row['FileName'])
            return

        cdst =  cv2.cvtColor(edges,  cv2.COLOR_GRAY2BGR)
        angles_deg = []

        # Able to work directly with the normal line because
        # theta is the same as the angle between the line and the y-axis
        # This works because you rotate both the line and the axis you compare it to by 90 degrees

        for line in lines:
            # The Origin is in the top left corner of the image
            # rho = distance from the origin to the line
            # theta = counter-clockwise angle from the x-axis to normal of the line [0, π]
            rho, theta = line[0]

            # When rho is negative, we need to flip the angle along the x-axis
            if rho < 0:
                theta = (theta + np.pi)

            angle = math.degrees(theta)

            # If the angle is close to 360 degrees, assume it's a vertical line
            if abs(angle - 360) < 1.1:
                angle = 0

            angles_deg.append(angle)

            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(cdst, pt1, pt2, (0, 0, 255), 1,  cv2.LINE_AA)

        angles_deg = sorted(angles_deg)

        # Calculate angle between two lines
        if len(angles_deg) >= 2:
            angle_between_lines = abs(angles_deg[0] - angles_deg[len(angles_deg) - 1])

            # Want the acute angle
            if (angle_between_lines > 180):
                angle_between_lines = 360 - angle_between_lines
            predicted_angles.append(angle_between_lines)

    # Draw the lines
    # cv2.imshow('Detected Lines', cdst)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
            
    # Calculate errors
    errors = np.abs(np.subtract(actual_angles, predicted_angles))
    total_error = np.sum(errors)

    combined_results = np.c_[actual_angles, predicted_angles, errors]

    # Create dataframe to show results
    results = pd.DataFrame(combined_results, index=[filename.strip('.png') for filename in task1_data.loc[:,'FileName']], columns=['Actual', 'Predicted', 'Error'])
    
    print(results)
    print(f"\nTotal error: {total_error}")

    return(total_error)

def testTask2(iconDir, testDir):
    # assume that test folder name has a directory annotations with a list of csv files
    # load train images from iconDir and for each image from testDir, match it with each class from the iconDir to find the best match
    # For each predicted class, check accuracy with the annotations
    # Check and calculate the Intersection Over Union (IoU) score
    # based on the IoU determine accuracy, TruePositives, FalsePositives, FalseNegatives
    return (Acc,TPR,FPR,FNR)


def testTask3(iconFolderName, testFolderName):
    # assume that test folder name has a directory annotations with a list of csv files
    # load train images from iconDir and for each image from testDir, match it with each class from the iconDir to find the best match
    # For each predicted class, check accuracy with the annotations
    # Check and calculate the Intersection Over Union (IoU) score
    # based on the IoU determine accuracy, TruePositives, FalsePositives, FalseNegatives
    return (Acc,TPR,FPR,FNR)


# if __name__ == "__main__":

#     # parsing the command line path to directories and invoking the test scripts for each task
#     parser = argparse.ArgumentParser("Data Parser")
#     parser.add_argument("--Task1Dataset", help="Provide a folder that contains the Task 1 Dataset.", type=str, required=False)
#     parser.add_argument("--IconDataset", help="Provide a folder that contains the Icon Dataset for Task2 and Task3.", type=str, required=False)
#     parser.add_argument("--Task2Dataset", help="Provide a folder that contains the Task 2 test Dataset.", type=str, required=False)
#     parser.add_argument("--Task3Dataset", help="Provide a folder that contains the Task 3 test Dataset.", type=str, required=False)
#     args = parser.parse_args()
#     if(args.Task1Dataset!=None):
#         # This dataset has a list of png files and a txt file that has annotations of filenames and angle
#         testTask1(args.Task1Dataset)
#     if(args.IconDataset!=None and args.Task2Dataset!=None):
#         # The Icon dataset has a directory that contains the icon image for each file
#         # The Task2 dataset directory has two directories, an annotation directory that contains the annotation and a png directory with list of images 
#         testTask2(args.IconDataset,args.Task2Dataset)
#     if(args.IconDataset!=None and args.Task3Dataset!=None):
#         # The Icon dataset directory contains an icon image for each file
#         # The Task3 dataset has two directories, an annotation directory that contains the annotation and a png directory with list of images 
#         testTask3(args.IconDataset,args.Task3Dataset)

testTask1('Task1Dataset')
# testCanny('Task1Dataset')
