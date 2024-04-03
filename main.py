import math
import numpy as np
import cv2
import pandas as pd
from matplotlib import pyplot as plt
import os

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
    # Blur the image and then apply Sobel in the x & y directions???

    # Apply Scharr filters to image
    gx = cv2.Scharr(image, cv2.CV_16S, 1, 0)
    gy = cv2.Scharr(image, cv2.CV_16S, 0, 1)

    # Calculate gradient magnitude and direction
    g = np.hypot(gx, gy)
    theta = np.arctan2(gy, gx)
    g = (g / g.max()) * 255

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
    # For each pixel check if it's a local maximum (in its direction)
    for row in range(0, g.shape[0]):
        for column in range(0, g.shape[1]):
            angle = theta[row][column]
            r_angle = round_angle(angle)

            if pixel_suppressed(g, r_angle, row, column) == False:
                suppress[row][column] = g[row][column]


    # Hysteresis Thresholding to remove isolated weak edges
    mask = np.zeros_like(suppress)

    # Boolean array, True if the pixel is a weak/strong edge
    strong_edges = (suppress >= t_high)
    weak_edges = (suppress >= t_low)

    # Each pixel is labeled with a number corresponding to its connected component
    # uint8 to convert Boolean array to a 0/255 array
    num_labels, labels = cv2.connectedComponents(np.uint8(weak_edges))

    # Look at each connected component and determine if it should be a strong edge
    # Starting at 1 because 0 is the background
    for label in range(1, num_labels):
        # Boolean array of pixels in the current component
        component = (labels == label)

        componentContainsStrongEdge = np.any(component & strong_edges)
        if componentContainsStrongEdge:
            mask[component] = 255

    return mask

    # cv2.imshow("suppressed", suppress)
    # cv2.imshow("a", a)
    # cv2.imshow("output", output)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def testCanny(folder_name):
    image =  cv2.imread(folder_name + '/' + 'image9.png', cv2.IMREAD_GRAYSCALE)
    canny(image, 50, 200)

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
        edges = canny(image, 50, 200)

        # lines = cv2.HoughLines(edges, 1, np.pi / 180, 90, None, 0, 0)
        # 1 degree = pi / 180
        lines = HoughLines(edges, 1, math.radians(1), 90)

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
        else:
            print(row['FileName'])

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

def build_gaussian_pyramid(image, downsamples):
    G = image.copy()
    pyramid = [G]
    for i in range(downsamples):
        # TODO: Downsample at a slower rate like 10% - image = cv2.resize(image, None, fx=0.9, fy=0.9, interpolation=cv2.INTER_AREA)
        # TODO: Start the template at a smaller scale???
        G = cv2.pyrDown(G)
        pyramid.append(G)
    return pyramid

def build_laplacian_pyramid(gp):
    lp = []
    pyr_height = len(gp)
    for i in range(pyr_height - 1, 0, -1):
        G = cv2.pyrUp(gp[i])
        L = cv2.subtract(gp[i - 1], G)
        lp.append(L)
    return lp

# Calculate correlation for given patch
def correlation(template, img, offset_x, offset_y):
    correlation = []
    for y in range(0, template.shape[0]):
        for x in range(0, template.shape[1]):
            template_I = template[y][x]
            img_I = img[y + offset_y][x + offset_x]
            correlation.append(template_I * img_I)
    return np.sum(correlation)

def matchTemplate(lp_test, lp_template, method):
    for test_index in range(0, len(lp_test)):
        test_img = lp_test[test_index]
        for template_index in range(0, len(lp_template)):
            template = lp_template[template_index]
    return

def evaluate_predictions(annotations, predicted_icons):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # Iterate over each predicted class
    for prediction in predicted_icons:
        icon_index = prediction[0]
        
        # Find the row in annotations that matches the index of the predicted icon
        annotation = annotations[annotations['classname'].str.startswith(icon_index)]

        # If there is no annotation for the predicted class, it is a false positive
        if annotation.empty:
            print(f"Predicted icon {icon_index} but it doesn't exist in the image")
            false_positives += 1
        else:
            # Get the coordinates of the predicted bounding box
            predicted_top = prediction[1]
            predicted_left = prediction[2]
            predicted_bottom = prediction[3]
            predicted_right = prediction[4]

            # Get the coordinates of the annotated bounding box
            annotated_top = annotation['top'].values[0]
            annotated_left = annotation['left'].values[0]
            annotated_bottom = annotation['bottom'].values[0]
            annotated_right = annotation['right'].values[0]

            # Calculate the intersection over union (IoU) score
            intersection_area = max(0, min(predicted_bottom, annotated_bottom) - max(predicted_top, annotated_top)) * max(0, min(predicted_right, annotated_right) - max(predicted_left, annotated_left))
            predicted_area = (predicted_bottom - predicted_top) * (predicted_right - predicted_left)
            annotated_area = (annotated_bottom - annotated_top) * (annotated_right - annotated_left)
            union_area = predicted_area + annotated_area - intersection_area
            iou = intersection_area / union_area

            print(f"Predicted icon {icon_index} with IoU score {iou}")

            # If the IoU score is above a threshold, it is a true positive
            # Because the predicted bounding box is considered a match to the annotated bounding box
            if iou >= 0.5:
                true_positives += 1
            else:
                false_positives += 1

    for _, annotation in annotations.iterrows():
        icon_index = annotation['classname'].split('-')[0]
        matching_predicitons = [x[0] == icon_index for x in predicted_icons]
        prediction_missing = sum(matching_predicitons) == 0

        # Checking if the array contains just false
        if prediction_missing:
            # Failed to predict an icon that exists in the image
            false_negatives += 1
            print(f"Failed to predict icon {icon_index}")

    return (true_positives, false_positives, false_negatives)

def testTask2(iconDir, testDir):
    # assume that test folder name has a directory annotations with a list of csv files
    # load train images from iconDir and for each image from testDir, match it with each class from the iconDir to find the best match
    # For each predicted class, check accuracy with the annotations
    # Check and calculate the Intersection Over Union (IoU) score
    # based on the IoU determine acceracy, TruePositives, FalsePositives, FalseNegatives
    # return (Acc,TPR,FPR,FNR)

    # Retrieve all the icons
    icon_folder = './IconDataset/png'
    icons = []
    for filename in os.listdir(icon_folder):
        icon_path = os.path.join(icon_folder, filename)
        # print(icon_path)
        # TODO: Color???
        icon = cv2.imread(icon_path, cv2.IMREAD_GRAYSCALE)
        blurred_icon = cv2.GaussianBlur(icon, ksize=(5,5), sigmaX=0)
        icons.append(blurred_icon)

    image_folder = './Task2Dataset/images'
    images = []
    image_names = os.listdir(image_folder)
    # Sorts on the number at the end of the filename
    sorted_image_names = sorted(image_names, key=lambda x: int(x.split('_')[2].split('.')[0]))
    for filename in sorted_image_names:
        image_path = os.path.join(image_folder, filename)
        # TODO: Color???
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        blurred_image = cv2.GaussianBlur(image, ksize=(5,5), sigmaX=0)
        images.append(blurred_image)

    # methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
    # 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    method = ['cv2.TM_CCOEFF_NORMED']
    method = eval(method[0])

    overall_TPs = 0
    overall_FPs = 0
    overall_FNs = 0

    for image_index, image in enumerate(images):
        # icon index, top, left, bottom, right
        predicted_icons = []

        display_image = image.copy()
        
        # Predict which icons are in the image
        for icon_index, icon in enumerate(icons):
            # score, location, template_index
            best_match = (0, 0, -1)
            templates = build_gaussian_pyramid(icon, 4)
            
            # # Create laplacian pyramid for template
            # gp_template = build_gaussian_pyramid(template, 5)
            # lp_template = build_laplacian_pyramid(gp_template)
            # # Create laplacian pyramid for test image
            # gp_test = build_gaussian_pyramid(img, 5)
            # lp_test = build_laplacian_pyramid(gp_test)

            # Multi-scale template matching, keeping only the best match
            for idx, templ in enumerate(templates):
                result = cv2.matchTemplate(image, templ, method)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)

                if best_match[0] < max_val:
                    # Update best_match
                    best_match = (max_val, max_loc, idx) 

            # Thresholding to prevent false positives
            if (best_match[0] < 0.95):
                continue

            # TODO: Understand how the max_loc from minMaxLoc is used to get the location on the original image
            print(f"Score for icon {icon_index + 1} = {best_match[0]} with template {best_match[2]} @ {best_match[1]}")

            best_template = templates[best_match[2]]
            w, h = best_template.shape[::-1]
            top_left = best_match[1]
            bottom_right = (top_left[0] + w, top_left[1] + h)

            print(top_left, bottom_right)
            print(best_match)
            print("")

            # Add the predicted icon
            # Do we need to know the template index? -> yes when scaling back up?
            str_icon_index = f"{icon_index + 1:02d}" # Padding with 0s to match the annotations
            predicted_icons.append([str_icon_index, *top_left, *bottom_right])
            
            # Bounding box
            cv2.rectangle(display_image, top_left, bottom_right, 0, 2)
            # Plots
            # res = cv2.matchTemplate(image, best_template, method)
            # plt.subplot(121),plt.imshow(res,cmap = 'gray')
            # plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
            # plt.subplot(122),plt.imshow(display_image,cmap = 'gray')
            # plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
            # plt.suptitle('Template ' + str(best_match[2]))
            # plt.show()

        # Evaluate the predicted icons
        annotations = pd.read_csv(f'./Task2Dataset/annotations/test_image_{image_index + 1}.csv')
        (true_positives, false_positives, false_negatives) = evaluate_predictions(annotations, predicted_icons)

        overall_TPs += true_positives
        overall_FPs += false_positives
        overall_FNs += false_negatives

        # Show all the detected icons in the current image
        # plt.imshow(display_image, cmap='gray')
        # plt.title(f'Image {image_index} - Detected Icons')
        # plt.xticks([]), plt.yticks([])
        # plt.show()

    # Evaluate the performance over all images
    print(f"Overall TPs: {overall_TPs}, Overall FPs: {overall_FPs}, Overall FNs: {overall_FNs}")

    # How often it detected an object when the object was not there
    # false_positive_rate = false_positives / (false_positives + true_negatives)
    # false_positive_rate = false_positives / (false_positives + true_positives)
    # false_negative_rate = false_negatives / (false_negatives + true_positives)
    true_positive_rate = true_positives / (true_positives + false_negatives)

    accuracy = overall_TPs / (overall_TPs + overall_FPs + overall_FNs)
    precision = overall_TPs / (overall_TPs + overall_FPs)
    recall = overall_TPs / (overall_TPs + overall_FNs)
    # Return the results
    print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, True Positive Rate: {true_positive_rate}")

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

# testTask1('Task1Dataset')
# testCanny('Task1Dataset')
testTask2('ah', 'ah')
