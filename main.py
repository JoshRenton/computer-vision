import math
import numpy as np
import cv2
import pandas as pd
from matplotlib import pyplot as plt
import os
import copy
import argparse

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
    lows = {}
    highs = {}

    # Get the indices of all edges in the edge array
    # The index is the coordinate of the edge since the edge array is the same size as the image
    edge_coords = np.asarray(edges == 255).nonzero()
    threshold = len(edge_coords[0]) * 0.15

    # Precalculate sin and cos values for each theta value
    theta_sins = np.sin(theta_values)
    theta_coss = np.cos(theta_values)

    # For each edge, calculate rho for each value of theta
    for y,x in zip(edge_coords[0], edge_coords[1]):
        for t_index in range(len(theta_values)):
            rho = x * theta_coss[t_index] + y * theta_sins[t_index]

            # Get the index of the closest rho value below the actual calculated rho
            r_index = np.searchsorted(rho_values, rho, side='left') - 1
            # Increment the vote for this cell
            accumulator[t_index, r_index] += 1

            # Keep track of highest and lowest x-coordinates of points that vote for each line
            if (t_index, r_index) in lows:
                if x < lows.get((t_index, r_index))[0]:
                    lows[(t_index, r_index)] = (x, y)
                # Defer to y-coordinate if x is equal for vertical lines
                elif x == lows.get((t_index, r_index))[0] and y < lows.get((t_index, r_index))[1]:
                    lows[(t_index, r_index)] = (x, y)
            else:
                lows[(t_index, r_index)] = (x, y)

            if (t_index, r_index) in highs:
                if x > highs.get((t_index, r_index))[0]:
                    highs[(t_index, r_index)] = (x, y)
                elif x == highs.get((t_index, r_index))[0] and y > highs.get((t_index, r_index))[1]:
                    highs[(t_index, r_index)] = (x, y)
            else:
                highs[(t_index, r_index)] = (x, y)

    index_to_votes = {}
    theta_indices, rho_indices = np.where(accumulator > threshold)
    for theta_index, rho_index in zip(theta_indices, rho_indices):
        index_to_votes[(theta_index, rho_index)] = accumulator[theta_index, rho_index]

    # sort the dict by the vote count
    index_to_votes = dict(sorted(index_to_votes.items(), key=lambda item: item[1], reverse=True))

    # Generate polar coordinates from the the top 2 cells in the dict
    polar_coordinates = []
    final_indices = []
    for theta_index, rho_index in index_to_votes.keys():
        theta = theta_values[theta_index]
        rho = rho_values[rho_index]

        if len(polar_coordinates) == 0:
            polar_coordinates.append([rho, theta])
            final_indices.append((rho_index, theta_index))
        elif abs(polar_coordinates[0][1] - theta) > math.radians(1):
            polar_coordinates.append([rho, theta])
            final_indices.append((rho_index, theta_index))
            break

    polar_coordinates = np.array(polar_coordinates)

    extremes = []
    for r_index, t_index in final_indices:
        extremes.append((lows[(t_index, r_index)], highs[(t_index, r_index)]))
    return (polar_coordinates, extremes)

# Round angle to nearest 45 degrees
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

# Check whether a pixel shold be suppressed
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

# Canny edge detection implementation
def canny(image, t_low, t_high):
    # Blur the image and then apply Sobel in the x & y directions???

    # Apply Scharr filters to image
    gx = cv2.Sobel(image, cv2.CV_16S, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_16S, 0, 1, ksize=3)

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
    strong_edge_pixels = (suppress >= t_high)
    pixels_above_t_low = (suppress >= t_low)

    # Each pixel is labeled with a number corresponding to its connected component
    # uint8 to convert Boolean array to a 0/255 array
    num_labels, labels = cv2.connectedComponents(np.uint8(pixels_above_t_low))

    # Look at each connected component and determine if it should be a strong edge
    # Starting at 1 because 0 is the background
    for label in range(1, num_labels):
        # Boolean array of pixels in the current component
        component = (labels == label)

        componentContainsStrongEdge = np.any(component & strong_edge_pixels)
        if componentContainsStrongEdge:
            mask[component] = 255

    return mask

# Return intersection between lines defined by a pair of points
def intersection(pts1, pts2):
    pt1, pt2 = pts1
    pt3, pt4 = pts2
    
    x1, y1 = pt1
    x2, y2 = pt2
    x3, y3 = pt3
    x4, y4 = pt4

    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    if denominator == 0:
        return (-1, -1)

    Px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
    Py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator

    return (Px, Py)

# Return absolute distance between 2 tuple points
def dist(t1, t2):
    x_diff = abs(t1[0] - t2[0])
    y_diff = abs(t1[1] - t2[1])
    return x_diff + y_diff

# Figure out the extremes where the lines connect
def align_lines(line1, line2):
    isct = intersection(line1, line2)

    # Find the two closest points (where the lines connect)
    diff1 = [dist(line1[i], isct) for i in range(0, len(line1))]
    diff2 = [dist(line2[i], isct) for i in range(0, len(line2))]

    # Make the first point of each line be the connecting point
    aligned_line1 = [x for _, x in sorted(zip(diff1, line1))]
    aligned_line2 = [x for _, x in sorted(zip(diff2, line2))]

    return aligned_line1, aligned_line2

def testTask1(folder_name):
    # Read in data
    task1_data = pd.read_csv(folder_name+"/list.txt")
    task1_data = task1_data.reset_index()

    actual_angles = []
    predicted_angles = []

    # Iterate through all images
    for index, row in task1_data.iterrows():
        image =  cv2.imread(folder_name + '/' + row['FileName'], cv2.IMREAD_GRAYSCALE)
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        actual_angles.append(row['AngleInDegrees'])
        edges = canny(image, 50, 200)

        # 1 degree = pi / 180
        lines, extremes = HoughLines(edges, 1, math.radians(1), 90)

        if lines is None:
            print(row['FileName'])
            return

        cdst =  cv2.cvtColor(edges,  cv2.COLOR_GRAY2BGR)
        angles_deg = []
        pts = []

        # Able to work directly with the normal line because
        # theta is the same as the angle between the line and the y-axis
        # This works because you rotate both the line and the axis you compare it to by 90 degrees

        for line in lines:
            # The Origin is in the top left corner of the image
            # rho = distance from the origin to the line
            # theta = counter-clockwise angle from the x-axis to normal of the line [0, π]

            # rho, theta = line[0]
            rho, theta = line
            rho = abs(rho)

            # When rho is negative, we need to flip the angle along the x-axis
            if rho < 0:
                theta = (theta + np.pi)

            angle = math.degrees(theta)
            print("rho:", rho, "\t theta:", angle)

            # If the angle is close to 360 degrees, assume it's a vertical line
            if abs(angle - 360) < 1.1:
                angle = 0

            angles_deg.append(angle)

        extremes = [ex for _, ex in sorted(zip(angles_deg, extremes))]
        angles_deg = sorted(angles_deg)

        detected_lines = np.zeros_like(image)
        detected_lines = cv2.cvtColor(detected_lines, cv2.COLOR_GRAY2BGR)

        # cv2.line(detected_lines, extremes[0][0], extremes[0][1], (0, 0, 255), 1,  cv2.LINE_AA)
        # cv2.line(detected_lines, extremes[len(extremes) - 1][0], extremes[len(extremes) - 1][1], (0, 0, 255), 1,  cv2.LINE_AA)
        cv2.line(cdst, extremes[0][0], extremes[0][1], (0, 0, 255), 1,  cv2.LINE_AA)
        cv2.line(cdst, extremes[len(extremes) - 1][0], extremes[len(extremes) - 1][1], (0, 0, 255), 1,  cv2.LINE_AA)

        line1 = extremes[0]
        line2 = extremes[len(extremes) - 1]

        line1, line2 = align_lines(line1, line2)

        x_diff1 = line1[1][0] - line1[0][0]
        x_diff2 = line2[1][0] - line2[0][0]
        y_diff1 = line1[1][1] - line1[0][1]
        y_diff2 = line2[1][1] - line2[0][1]

        # Line segment is facing left from connection point
        if (x_diff1 < 0):
            angles_deg[0] += 180
        if (x_diff2 < 0):
            angles_deg[len(angles_deg) - 1] += 180

        # Check for direction of vertical line from connection point
        if (abs(x_diff1) < 1.5 and y_diff1 > 0):
            angles_deg[0] = 180
        elif (abs(x_diff1) < 1.5 and y_diff1 < 0):
            angles_deg[0] = 0
        if (abs(x_diff2) < 1.5 and y_diff2 > 0):
            angles_deg[len(angles_deg) - 1] = 180
        elif (abs(x_diff2) < 1.5 and y_diff2 < 0):
            angles_deg[len(angles_deg) - 1] = 0

        # Calculate angle between two lines
        if len(angles_deg) >= 2:
            angle_between_lines = abs(angles_deg[0] - angles_deg[len(angles_deg) - 1])

            # Want the acute angle
            if (angle_between_lines > 180):
                angle_between_lines = 360 - angle_between_lines
            predicted_angles.append(np.round(angle_between_lines))
        else:
            print(row['FileName'])

        # cv2.imwrite('temp/' + row['FileName'], detected_lines)

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


# -----------------------
#         TASK 2
# -----------------------
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

def normalize_img(img):
    norm_img = img - np.mean(img)
    return norm_img

# Calculate correlation for given patch
def correlation(template, img, offset_x, offset_y):
    img_patch = img[offset_y:offset_y + template.shape[0], offset_x:offset_x + template.shape[1]]
    img_std = np.std(img_patch)
    if img_std == 0:
        return -1
    correlation = img_patch * template
    correlation = correlation / (img_std * np.std(template))
    return np.sum(correlation) / (template.shape[0] * template.shape[1])

def matchTemplate(img, template):
    if template.shape[1] > img.shape[1] or template.shape[0] > img.shape[0]:
        return np.array([0])
    
    norm_template = normalize_img(template)
    corr_scores = np.zeros((img.shape[0] - template.shape[0] + 1, img.shape[1] - template.shape[1] + 1))
    for y in range(0, img.shape[0] - template.shape[0] + 1):
        for x in range(0, img.shape[1] - template.shape[1] + 1):
            corr = correlation(norm_template, img, x, y)
            corr_scores[y][x] = corr

    return corr_scores

def matchTemplateWithCoords(img, template, coords):
    if template.shape[1] > img.shape[1] or template.shape[0] > img.shape[0]:
        return None, None
    
    norm_template = normalize_img(template)
    corr_scores = []
    corr_coords = []

    test_x, test_y = coords

    for y in range(test_y - 10, test_y + 10):
        if y >= 0 and y < img.shape[0] - template.shape[0] + 1:
            for x in range(test_x - 10, test_x + 10):
                if x >= 0 and x < img.shape[1] - template.shape[1] + 1:
                    corr = correlation(norm_template, img, x, y)
                    corr_scores.append(corr)
                    corr_coords.append((x, y))

    if len(corr_scores) == 0:
        return None, None

    return corr_scores, corr_coords

def minMaxLoc(corr):
    min_value = np.min(corr)
    max_value = np.max(corr)
    min_index = np.unravel_index(np.argmin(corr, axis=None), corr.shape)
    max_index = np.unravel_index(np.argmax(corr, axis=None), corr.shape)
    # Have to flip because of opencv reversing (x,y) coordinates
    return min_value, max_value, np.flip(min_index), np.flip(max_index)

def evaluate_predictions(annotations, predicted_icons):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # Iterate over each predicted class
    for prediction in predicted_icons:
        icon_name = prediction[0]
        
        # Find the row in annotations that matches the index of the predicted icon
        annotation = annotations[annotations['classname'] == icon_name]

        # If there is no annotation for the predicted class, it is a false positive
        if annotation.empty:
            print(f"Predicted {icon_name} but it doesn't exist in the image")
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

            print(f"Predicted {icon_name} with IoU score {iou}")

            # If the IoU score is above a threshold, it is a true positive
            # Because the predicted bounding box is considered a match to the annotated bounding box
            if iou >= 0.5:
                true_positives += 1
            else:
                false_positives += 1

    for _, annotation in annotations.iterrows():
        icon_name = annotation['classname']
        prediction_missing = not any(prediction[0] == icon_name for prediction in predicted_icons)

        # Checking if the array contains just false
        if prediction_missing:
            # Failed to predict an icon that exists in the image
            false_negatives += 1
            print(f"Failed to predict icon {icon_name}")

    return (true_positives, false_positives, false_negatives)

def searchImage(img, test_icons, test_coords, threshold, pyr_depth):
    predicted_icons = []
    index = 0

    for icon_name, icon in test_icons:
        best_match = (0, 0, -1)
        # Currently creating pyramid every time
        templates = build_gaussian_pyramid(icon, pyr_depth)
        r_icon = cv2.resize(icon, (int(icon.shape[0] * 0.75), int(icon.shape[1] * 0.75)))
        r_templates = build_gaussian_pyramid(r_icon, pyr_depth)
        ori_templates = copy.deepcopy(templates)
        templates = []
        for i in range(0, len(ori_templates)):
            templates.append(ori_templates[i])
            templates.append(r_templates[i])
            
        # Multi-scale template matching, keeping only the best match
        # Only do full search of all templates on smallest image size
        if test_coords == None:
            for idx, templ in enumerate(templates):
                result = matchTemplate(img, templ)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)

                if best_match[0] < max_val:
                    # Update best_match
                    best_match = (max_val, max_loc, idx)
        else:
            # Search for icons identified in previous layer around the matching upscaled coords
            for idx, templ in enumerate(templates):
                result, result_coords = matchTemplateWithCoords(img, templ, test_coords[index])
                if result == None or result_coords == None:
                    continue
                max_index = np.argmax(result)
                max_val = result[max_index]
                max_loc = result_coords[max_index]

                if best_match[0] < max_val:
                    # Update best_match
                    best_match = (max_val, max_loc, idx)

            index += 1

        # Thresholding to prevent false positives
        # Increased threshold after lowest resolution search
        # if test_coords == None:
        #     if (best_match[0] < 0.85):
        #         continue
        # else:
        #     if (best_match[0] < 0.95):
        #         continue
            
        if (best_match[0] < threshold):
            continue

        best_template = templates[best_match[2]]
        match_size = best_template.shape[::-1]

        predicted_icons.append([icon_name, best_match, match_size])
    
    return predicted_icons


def testTask2(iconDir, testDir):
    # Retrieve all the icons (train images)
    icon_folder = f'./{iconDir}/png'
    icons = []
    icon_names = []
    for filename in os.listdir(icon_folder):
        # Remove the leading zero and the file extension
        icon_names.append(filename[1:].rsplit(".", 1)[0])

        icon_path = os.path.join(icon_folder, filename)
        icon = cv2.imread(icon_path, cv2.IMREAD_GRAYSCALE)
        blurred_icon = cv2.GaussianBlur(icon, ksize=(5,5), sigmaX=0)
        icons.append(blurred_icon)

    image_folder = f'./{testDir}/images'
    images = []
    image_names = os.listdir(image_folder)
    # Sorts on the number at the end of the filename
    sorted_image_names = sorted(image_names, key=lambda x: int(x.split('_')[2].split('.')[0]))
    for filename in sorted_image_names:
        image_path = os.path.join(image_folder, filename)
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
        # icon name, top, left, bottom, right
        test_icons = zip(icon_names, icons)
        test_coords = None

        # Create gaussian pyramid of test_image
        img_pyr = build_gaussian_pyramid(image, 3)

        threshold = 0.85
        pyr_depth = 6
        # Start at smallest resolution
        for layer in range(len(img_pyr) - 1, -1, -1):

            img = img_pyr[layer]
            norm_img = normalize_img(img)
            display_image = img.copy()
            
            # Predict which icons are in the image
            predicted_icons = searchImage(norm_img, test_icons, test_coords, threshold, pyr_depth)
            test_icons = []
            test_coords = []
            predictions = []

            for match in predicted_icons:
                icon_name = match[0]
                best_match = match[1]
                match_size = match[2]

                icon_index = icon_names.index(icon_name)
                test_icons.append((icon_name, icons[icon_index]))
                test_coords.append((best_match[1][0] * 2, best_match[1][1] * 2))

                # TODO: Understand how the max_loc from minMaxLoc is used to get the location on the original image
                print(f"Score for {icon_name} = {best_match[0]} with template {best_match[2]} @ {best_match[1]}")

                w, h = match_size
                top_left = best_match[1]
                bottom_right = (top_left[0] + w, top_left[1] + h)

                print(top_left, bottom_right)
                print(best_match)
                print("")
                
                # Bounding box
                cv2.rectangle(display_image, top_left, bottom_right, 0, 2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottom_left = (top_left[0], top_left[1] - 5)  # Position the text at the bottom left of the rectangle
                cv2.putText(display_image, icon_name, bottom_left, font, 0.5, 0, 1, cv2.LINE_AA)
                # Plots
                # res = cv2.matchTemplate(image, best_template, method)
                # plt.subplot(121),plt.imshow(res,cmap = 'gray')
                # plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
                # plt.subplot(122),plt.imshow(display_image,cmap = 'gray')
                # plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
                # plt.suptitle('Template ' + str(best_match[2]))
                # plt.show()

                predictions.append([icon_name, top_left[0], top_left[1], bottom_right[0], bottom_right[1]])

            if layer == 0:
                # Evaluate the predicted icons
                annotations = pd.read_csv(f'./{testDir}/annotations/test_image_{image_index + 1}.csv')
                (true_positives, false_positives, false_negatives) = evaluate_predictions(annotations, predictions)

                overall_TPs += true_positives
                overall_FPs += false_positives
                overall_FNs += false_negatives

                # Show all the detected icons in the current image
                plt.imshow(display_image, cmap='gray')
                plt.title(f'Image {image_index + 1}')
                plt.xticks([]), plt.yticks([])
                plt.show()

            threshold += 0.03
            pyr_depth -= 1

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

    # TODO:
    # based on the IoU determine acceracy, TruePositives, FalsePositives, FalseNegatives
    # return (Acc,TPR,FPR,FNR)

def testTask3(iconFolderName, testFolderName):
    # assume that test folder name has a directory annotations with a list of csv files
    # load train images from iconDir and for each image from testDir, match it with each class from the iconDir to find the best match
    # For each predicted class, check accuracy with the annotations
    # Check and calculate the Intersection Over Union (IoU) score
    # based on the IoU determine accuracy, TruePositives, FalsePositives, FalseNegatives
    return (Acc,TPR,FPR,FNR)


if __name__ == "__main__":
    # parsing the command line path to directories and invoking the test scripts for each task
    parser = argparse.ArgumentParser("Data Parser")
    parser.add_argument("--Task1Dataset", help="Provide a folder that contains the Task 1 Dataset.", type=str, required=False)
    parser.add_argument("--IconDataset", help="Provide a folder that contains the Icon Dataset for Task2 and Task3.", type=str, required=False)
    parser.add_argument("--Task2Dataset", help="Provide a folder that contains the Task 2 test Dataset.", type=str, required=False)
    parser.add_argument("--Task3Dataset", help="Provide a folder that contains the Task 3 test Dataset.", type=str, required=False)
    args = parser.parse_args()
    if(args.Task1Dataset!=None):
        # This dataset has a list of png files and a txt file that has annotations of filenames and angle
        testTask1(args.Task1Dataset)
    if(args.IconDataset!=None and args.Task2Dataset!=None):
        # The Icon dataset has a directory that contains the icon image for each file
        # The Task2 dataset directory has two directories, an annotation directory that contains the annotation and a png directory with list of images 
        testTask2(args.IconDataset,args.Task2Dataset)
    if(args.IconDataset!=None and args.Task3Dataset!=None):
        # The Icon dataset directory contains an icon image for each file
        # The Task3 dataset has two directories, an annotation directory that contains the annotation and a png directory with list of images 
        testTask3(args.IconDataset,args.Task3Dataset)

# testTask1('Task1Dataset')
# testTask2('IconDataset', 'Task2Datset')