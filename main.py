import math
import numpy as np
import cv2
import pandas as pd
from matplotlib import pyplot as plt
import os
import argparse
from datetime import datetime

def HoughLines(edges, rho_res, theta_res):
    """
    Parameters:
        edges (): 2D array representing the edges of an image
        rho_res (double): Resolution of rho in pixels
        theta_res (double): Resolution of theta in radians
 
    Returns:
        (polar_coordinates, extremes): The polar coordinates of the two best detected lines and the coordinates of their extremes
    """

    # edges is a 2D array with one cell per pixel of the original image
    # Each cell is a value between 0 and 255, with 255 being an edge

    # theta_res = pi / 180 = 1 degree, giving 180 theta values
    theta_values = np.arange(0, np.pi, theta_res)

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

    # Sorts the accumulator by vote count in descending order and returns the indices of the cells
    sorted_indicies = np.argsort(accumulator.flatten())[::-1]
    # Convert these indices back to 2D indices
    sorted_indicies = np.unravel_index(sorted_indicies, accumulator.shape)

    extremes = []
    polar_coordinates = []
    for t_index, r_index in zip(sorted_indicies[0], sorted_indicies[1]):
        theta = theta_values[t_index]
        rho = rho_values[r_index]

        if len(polar_coordinates) == 0:
            polar_coordinates.append([rho, theta])
            extremes.append((lows[(t_index, r_index)], highs[(t_index, r_index)]))
        # Looping till a valid 2nd best line is found
        elif abs(polar_coordinates[0][1] - theta) > math.radians(1):
            polar_coordinates.append([rho, theta])
            extremes.append((lows[(t_index, r_index)], highs[(t_index, r_index)]))
            break

    # polar_coordinates = np.array(polar_coordinates)
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
        actual_angles.append(row['AngleInDegrees'])
        edges = canny(image, 50, 200)

        # 1 degree = pi / 180 radians
        lines, extremes = HoughLines(edges, 1, math.radians(1))

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

            rho, theta = line
            rho = abs(rho)

            # # When rho is negative, we need to flip the angle along the x-axis
            # if rho < 0:
            #     theta = (theta + np.pi)

            angle = math.degrees(theta)
            # print("rho:", rho, "\t theta:", angle)

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

    # Calculate errors
    errors = np.abs(np.subtract(actual_angles, predicted_angles))
    total_error = np.sum(errors)

    combined_results = np.c_[actual_angles, predicted_angles, errors]

    # Create dataframe to show results
    results = pd.DataFrame(combined_results, index=[filename.strip('.png') for filename in task1_data.loc[:,'FileName']], columns=['Actual', 'Predicted', 'Error'])
    
    # print("")
    # print(results)
    # print(f"\nTotal error: {total_error}")

    return(total_error)


# -----------------------
#         TASK 2
# -----------------------
def build_gaussian_pyramid(image, downsamples):
    G = image.copy()
    pyramid = [G]

    for _ in range(downsamples):
        G = cv2.pyrDown(G) # 50% downscale
        pyramid.append(G)
    return pyramid

# Calculate ncc for all patches
def normalised_cross_correlation(patches, template):
    # patches = a 4D array

    # keepdims=True - the mean array will also be 4D so we can easily use it with the patches array
    # axis(2, 3) - the mean is calculated on the last 2 dimensions - the patch itself
    patch_means = np.mean(patches, axis=(2, 3), keepdims=True)
    patches_deviation = patches - patch_means
    template_deviation = template - np.mean(template)

    # Calculating the similarity score between the template and each patch, stored in a 2D array

    # Quicker, but normalises the sums rather than the whole template & patch
    # cross_correlation = np.sum(patches_deviation * template_deviation, axis=(2, 3))
    # denominator = np.sqrt(np.sum(patches_deviation**2, axis=(2,3)) * np.sum(template_deviation**2)) + 1e-6
    # ncc_scores = cross_correlation / denominator

    # Normalising the pixels in the patch & template before correlating
    normalised_patches = patches_deviation  / (1e-6 + np.sqrt(np.sum( patches_deviation**2, axis=(2,3), keepdims=True )))
    normalised_template = template_deviation / (1e-6 + np.sqrt(np.sum( template_deviation**2 )))
    ncc_scores = np.sum(normalised_patches * normalised_template, axis=(2, 3))

    # ncc_scores range from -1 to 1
    return ncc_scores

def matchTemplate(img, template):
    img_height, img_width = img.shape
    templ_height, templ_width = template.shape

    if templ_width > img_width or templ_height > img_height:
        return np.array([0])

    # Create a sliding window view of the image - generating all the patches from the image
    # patches = a 4D array -> A 2D array where each element is a patch of the image (with the same shape as the template)
    # The position of the patch in the 2D array is the position of the top-left corner of the patch in the image
    patches = np.lib.stride_tricks.sliding_window_view(img, template.shape)
    # NOTE: Would be quicker to pre-calculate these patches for the set template sizes, but it's quick enough

    return normalised_cross_correlation(patches, template)

def matchTemplateWithCoords(img, template, coords):
    img_height, img_width = img.shape
    templ_height, templ_width = template.shape

    if templ_width > img_width or templ_height > img_height:
        return None, None
    
    y = coords[1]
    x = coords[0]
    # Calculate the start and end indices for the y and x dimensions
    offset = 3
    y_start = max(0, y - offset)
    x_start = max(0, x - offset)
    y_end = min(img_height - templ_height + 1, y + offset)
    x_end = min(img_width  - templ_width  + 1, x + offset)

    patches = np.lib.stride_tricks.sliding_window_view(img, template.shape)
    patches = patches[y_start:y_end, x_start:x_end] # Select the relevant patches
    corr_scores = normalised_cross_correlation(patches, template)

    # Flatten corr_scores as the grid can't be used to get the coordinates 
    corr_scores = corr_scores.flatten().tolist()
    corr_coords = [(x, y) for y in range(y_start, y_end) for x in range(x_start, x_end)]

    if len(corr_scores) == 0:
        return None, None

    return corr_scores, corr_coords

def evaluate_predictions(annotations, predicted_icons):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    iou_sum = 0

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
            iou_sum += iou

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

    return (true_positives, false_positives, false_negatives, iou_sum)

def recursiveIconPrediction(img_pyr, threshold, icon_pyramids, icon_indicies, test_coords, pyr_depth, depth):
    img = img_pyr[depth]

    predicted_icons = []
    is_initial_search = test_coords == None

    # Generate icon predictions for this layer
    for index, icon_index in enumerate(icon_indicies):
        # score, location, template_index
        best_match = (0, 0, -1)

        # Only want to access the first 'pyr_depth' layers of the pyramid
        icon_templates = icon_pyramids[icon_index][:pyr_depth]

        # Multi-scale template matching - keeping only the best matching template for this icon
        if is_initial_search:
            # This top layer is the only layer that is searched for all icons
            for templ_index, templ in enumerate(icon_templates):
                result = matchTemplate(img, templ)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                if best_match[0] < max_val:
                    # Update best_match
                    best_match = (max_val, max_loc, templ_index)
        else:
            # Search for icons identified in previous layer around the matching upscaled coords
            for templ_index, templ in enumerate(icon_templates):
                result, result_coords = matchTemplateWithCoords(img, templ, test_coords[index])
                if result == None or result_coords == None:
                    continue
                max_index = np.argmax(result)
                max_val = result[max_index]
                max_loc = result_coords[max_index]

                if best_match[0] < max_val:
                    # Update best_match
                    best_match = (max_val, max_loc, templ_index)

        # Thresholding to prevent false positives
        if (best_match[0] < threshold):
            continue

        best_template = icon_templates[best_match[2]]
        match_size = best_template.shape[::-1]
        # print(f"Score for icon {icon_index} = {best_match[0]} with template {best_match[2]} @ {best_match[1]}")

        predicted_icons.append([icon_index, best_match, match_size])
    
    # Base case - return final predictions
    if depth == 0:
        final_predictions = []
        for prediction in predicted_icons:
            icon_index = prediction[0]
            best_match = prediction[1]
            match_size = prediction[2]

            w, h = match_size
            top_left = best_match[1]
            bottom_right = (top_left[0] + w, top_left[1] + h)

            # final_predictions.append([icon_index, top_left[0], top_left[1], bottom_right[0], bottom_right[1]])
            final_predictions.append([icon_index, *top_left, *bottom_right])
        return final_predictions

    # Otherwise use the predictions to update the search on the next layer down
    icon_indicies = []
    test_coords = []
    for prediction in predicted_icons:
        icon_index = prediction[0]
        best_match = prediction[1] # (score, location, template_index)
        match_size = prediction[2]

        icon_indicies.append(icon_index)    
        # Upscale the cell with the highest correlation score
        test_coords.append((best_match[1][0] * 2, best_match[1][1] * 2))

    return recursiveIconPrediction(img_pyr, threshold+0.06, icon_pyramids, icon_indicies, test_coords, pyr_depth-2, depth-1)

# Displaying all the predicted icons (and their bounding boxes) on the image
def show_predictions(image, image_index, predictions):
    display_image = image.copy() # Used for displaying the predicted icons

    # Generate a bounding box for each icon prediction
    for prediction in predictions:
        icon_name = prediction[0]
        top_left = (prediction[1], prediction[2])
        bottom_right = (prediction[3], prediction[4])

        cv2.rectangle(display_image, top_left, bottom_right, 0, 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottom_left = (top_left[0], top_left[1] - 5)  # Position the text at the bottom left of the rectangle
        cv2.putText(display_image, icon_name, bottom_left, font, 0.5, 0, 1, cv2.LINE_AA)

    # Show all the detected icons in the current image
    plt.imshow(display_image, cmap='gray')
    plt.title(f'Image {image_index + 1}')
    plt.xticks([]), plt.yticks([])
    plt.show()
    
    # Plots
    # res = cv2.matchTemplate(image, best_template, method)
    # plt.subplot(121),plt.imshow(res,cmap = 'gray')
    # plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122),plt.imshow(display_image,cmap = 'gray')
    # plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    # plt.suptitle('Template ' + str(best_match[2]))
    # plt.show()

def testTask2(iconDir, testDir):
    # Retrieve all the icons (train images)
    icon_folder = f'./{iconDir}/png'
    icon_names = []
    icon_pyramids = []
    for filename in os.listdir(icon_folder):
        # Remove the leading zero and the file extension
        icon_names.append(filename[1:].rsplit(".", 1)[0])

        icon_path = os.path.join(icon_folder, filename)
        icon = cv2.imread(icon_path, cv2.IMREAD_GRAYSCALE)

        templates = build_gaussian_pyramid(icon, 6)
        half_step = cv2.resize(icon, dsize=None, fx=0.75, fy=0.75)
        half_step_templates = build_gaussian_pyramid(half_step, 6)
        icon_pyr = []
        for i in range(0, len(templates)):
            icon_pyr.append(templates[i])
            icon_pyr.append(half_step_templates[i])
        icon_pyramids.append(icon_pyr) # 14 layers per icon

    # Print the size of the last element in the icon_pyramids list
    # print(icon_pyramids[-1][-1].shape)

    image_folder = f'./{testDir}/images'
    images = []
    image_names = os.listdir(image_folder)
    # Sorts on the number at the end of the filename
    sorted_image_names = sorted(image_names, key=lambda x: int(x.split('_')[2].split('.')[0]))
    for filename in sorted_image_names:
        image_path = os.path.join(image_folder, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        images.append(image)

    overall_TPs = 0
    overall_FPs = 0
    overall_FNs = 0
    overall_iou = 0
    time_taken = []

    icon_indicies = list(range(len(icon_names)))
    pyr_depth = len(icon_pyramids[0])
    threshold = 0.75

    for image_index, image in enumerate(images):
        print(f'Image {image_index + 1}')

        # Create gaussian pyramid of test_image
        img_pyr = build_gaussian_pyramid(image, 3)
        # Top layer with size 64x64

        # Predict which icons are in the image, by recursively searching each layer of the image pyramid - starting at the top
        # Searching only for the icons predicted in the previous layer at specific locations
        # final_predictions = (icon index, top, left, bottom, right)
        start=datetime.now()
        final_predictions = recursiveIconPrediction(img_pyr, threshold, icon_pyramids, icon_indicies, test_coords=None, pyr_depth=pyr_depth, depth=len(img_pyr)-1)
        elapsed_time = datetime.now() - start
        time_taken.append(elapsed_time.total_seconds())

        # Convert the icon index to the icon name
        final_predictions = [[icon_names[prediction[0]]] + prediction[1:] for prediction in final_predictions]

        # Evaluate the predicted icons
        annotations = pd.read_csv(f'./{testDir}/annotations/test_image_{image_index + 1}.csv')
        (true_positives, false_positives, false_negatives, iou_sum) = evaluate_predictions(annotations, final_predictions)

        # show_predictions(image, image_index, final_predictions)

        overall_TPs += true_positives
        overall_FPs += false_positives
        overall_FNs += false_negatives
        overall_iou += iou_sum

    # Evaluate the performance over all images
    print(f"Overall TPs: {overall_TPs}, Overall FPs: {overall_FPs}, Overall FNs: {overall_FNs}")
    print(f"Overall IoU: {overall_iou}")

    average_time = sum(time_taken) / len(time_taken)
    print(f"Average time taken: {average_time}")
    print(f'Overall runtime: {sum(time_taken)}')

    max_possible_icons = len(icon_names) * len(images)
    # TN = Correctly predicted an icon wasn't in the image
    overall_TNs =  max_possible_icons - overall_TPs - overall_FPs - overall_FNs

    TPR  = overall_TPs  / (overall_TPs + overall_FNs)
    FPR = overall_FPs / (overall_FPs + overall_TNs)
    if overall_FNs == 0:
        FNR = 0
    else:
        FNR = overall_FNs / (overall_FPs + overall_FNs)
    print(f'True Positive Rate: {TPR}')
    print(f'False Positive Rate: {FPR}')
    print(f'False Negative Rate: {FNR}')

    accuracy = overall_TPs / (overall_TPs + overall_FPs + overall_FNs)
    print(f'Accuracy: {accuracy}')

    # Return the results
    return (accuracy,TPR,FPR,FNR)

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