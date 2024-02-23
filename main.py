import math
import numpy as np
import cv2

def test():
    image =  cv2.imread('Task1Dataset/image4.png', cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(image, 50, 200, None, 3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 90, None, 0, 0)

    if lines is None:
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
    print("deg:", angles_deg)

    # Calculate angle between two lines
    if len(angles_deg) >= 2:
        angle_between_lines = abs(angles_deg[0] - angles_deg[len(angles_deg) - 1])

        # Want the acute angle
        if (angle_between_lines > 180):
            angle_between_lines = 360 - angle_between_lines
        print(f"Angle between two lines: {angle_between_lines} degrees")

    # Draw the lines
    cv2.imshow('Detected Lines', cdst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def testTask1(folderName):
    # assume that this folder name has a file list.txt that contains the annotation
    # task1Data = pd.read_csv(folderName+"/list.txt")
    # Write code to read in each image
    # Write code to process the image
    # Write your code to calculate the angle and obtain the result as a list predAngles
    # Calculate and provide the error in predicting the angle for each image

    lines = cv2.imread('Task1Dataset/image1.png', cv2.IMREAD_GRAYSCALE)
    dst = cv2.Canny(lines, 50, 200, None, 3)
    hough = cv2.HoughLines(dst, 1, np.pi / 180, 86, None, 0, 0)

    print(hough)

    cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)

    if hough is None:
        return
    radians = []
    for i in range (0, len(hough)):
        rho = hough[i][0][0]
        theta = hough[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        cv2.line(cdst, pt1, pt2, (0, 0, 255), 1, cv2.LINE_AA)
        print("theta: ", theta)
        # Maybe check dx
        if (math.isclose(theta, np.pi / 2, rel_tol=0.005)):
            radians.append(0)
        if (theta == 0):
            radians.append(np.pi / 2)
        elif (theta > (np.pi * 3/2) and theta < (np.pi * 2)):
            radians.append((np.pi * 2) + theta)
        elif (theta > (np.pi / 2) and theta < np.pi):
            radians.append(theta - (np.pi / 2))
        elif (theta > 0 and theta < (np.pi / 2)):
            radians.append((np.pi / 2) + theta)
        elif (theta > np.pi and theta < (np.pi * 3/2)):
            radians.append(theta - (np.pi / 2))
 
    for a in radians:
        print(a * (180 / np.pi))

    # unique_radians = np.unique(radians)
    radians = sorted(radians)
    angle = np.abs(radians[0] - radians[len(radians) - 1])
    if angle > np.pi:
        angle =  np.pi * 2 - angle

    print(angle * (180 / np.pi))

    cv2.imshow("edges", lines)
    cv2.imshow('hough', cdst)
    cv2.waitKey()
    
    # return(totalError)

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

test()
# testTask1('a')
