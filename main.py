import math
import numpy as np
import cv2
import pandas as pd


def HoughLines(edges, rho_res, theta_res, threshold):
    """
    Multiplies two numbers and returns the result.

    Parameters:
        edges (): 2D array representing the edges of an image
        rho_res (double): Resolution of rho in pixels
        theta_res (double): Resolution of theta in radians
        threshold (int): Minimum number of intersections to detect a line
 
    Returns:
        int: blah blah blah
    """

    # edges is a 2D array with one cell per pixel of the original image
    # Each cell is a value between 0 and 255, with 255 being an edge

    # TODO: Remove resolution parameters? Functions breaks if they're changed

    # theta_res = pi / 180 = 1 degree, giving 180 theta values
    theta_values = np.arange(0, np.pi, theta_res)

    # TODO: Not sure how you would get at most -diagonal_length
    # Max rho is the diagonal of the image (when theta = 45 degrees)
    diagonal_length = np.sqrt(edges.shape[0]**2 + edges.shape[1]**2)
    rho_values = np.arange(-diagonal_length, diagonal_length, rho_res)

    # Create an accumulator array to store the votes (in parameter space)
    accumulator = np.zeros([len(theta_values), len(rho_values)])

    # Get the indices of all edges in the edge array
    # The index is the coordinate of the edge since the edge array is the same size as the image
    edge_coords = np.asarray(edges == 255).nonzero()

    # Precalculate sin and cos values for each theta value
    theta_sins = np.sin(theta_values)
    theta_coss = np.cos(theta_values)

    # For each edge, calculate rho for each value of theta
    for x,y in zip(edge_coords[0], edge_coords[1]):
        for t in range(len(theta_values)):
            rho = x * theta_coss[t] + y * theta_sins[t]

            # Get the index of the closest rho value below the actual calculated rho
            r_index = np.searchsorted(rho_values, rho, side='left') - 1
            # Increment the vote for this cell
            accumulator[t, r_index] += 1

    # TODO: Look at hough_peaks to see if this function can return just one line

    # TODO: Need to rewrite these lines, copy pasta
    final_theta_index, final_rho_index = np.where(accumulator > threshold)
    final_rho = rho_values[final_rho_index]    
    final_theta = theta_values[final_theta_index]
    
    polar_coordinates = np.vstack([final_rho, final_theta]).T
    return polar_coordinates

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
        edges = cv2.Canny(image, 50, 200, None, 3)

        # lines = cv2.HoughLines(edges, 1, np.pi / 180, 90, None, 0, 0)
        # 1 degree = pi / 180
        lines = HoughLines(edges, 1, math.radians(1), 90)

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

            # rho, theta = line[0]
            rho, theta = line

            # When rho is negative, we need to flip the angle along the x-axis
            if rho < 0:
                theta = (theta + np.pi)

            angle = math.degrees(theta)
            print("rho:", rho, "\t theta:", angle)

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
    
    print("")
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
