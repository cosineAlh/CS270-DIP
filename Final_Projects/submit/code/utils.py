import numpy as np
import cv2
from scipy.ndimage import gaussian_filter, sobel
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def show(image):
    show_image = linear_normalize(image).astype(np.uint8)
    cv2.imshow('', show_image)
    cv2.waitKey()


def linear_normalize(image):
    # if len(image.shape) == 2:
    #     image_max = np.max(image)
    #     image_min = np.min(image)
    #     out_image = (image - image_min) / (image_max - image_min) * 255
    # else:
    #     out_image = np.zeros_like(image)
    #     for c in range(3):
    #         out_image[:, :, c] = linear_normalize(image[:, :, c])
    # return out_image
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def Canny(image):
    # if edge is coarse and discontinuous, may be problem in NMS
    # Gaussian smooth
    smoothed_image = gaussian_filter(image.astype(float), sigma=1)
    # smoothed_image = cv2.GaussianBlur(image, ksize=(11, 11), sigmaX=3)
    # show(smoothed_image)

    # Sobel Gradient Extract
    dx = sobel(smoothed_image, axis=1)
    # show(dx)
    dy = sobel(smoothed_image, axis=0)
    # show(dy)
    magnitude = np.sqrt(dx ** 2 + dy ** 2)
    # show(magnitude)
    angle = np.arctan2(dy, dx)      # arctan2(dy/dx) -> -pi ~ pi

    # NMS
    nms_magnitude = magnitude.copy()
    for i in range(1, magnitude.shape[0]-1):
        for j in range(1, magnitude.shape[1]-1):
            if 0 <= np.abs(np.pi - angle[i, j]) <= np.pi / 8 or \
                0 <= np.abs(- np.pi - angle[i, j]) <= np.pi / 8 or \
                    0 <= np.abs(0 - angle[i, j]) <= np.pi / 8:
                nms_magnitude[i, j] = 0 if magnitude[i, j] != max(magnitude[i, j - 1],
                                                                  magnitude[i, j],
                                                                  magnitude[i, j + 1]) else magnitude[i, j]
            elif 0 <= np.abs(- 3 * np.pi / 4 - angle[i, j]) <= np.pi / 8 or \
                    0 <= np.abs(np.pi / 4 - angle[i, j]) <= np.pi / 8:
                nms_magnitude[i, j] = 0 if magnitude[i, j] != max(magnitude[i - 1, j - 1],
                                                                  magnitude[i, j],
                                                                  magnitude[i + 1, j + 1]) else magnitude[i, j]
            elif 0 <= np.abs(- np.pi / 2 - angle[i, j]) <= np.pi / 8 or \
                    0 <= np.abs(np.pi / 2 - angle[i, j]) <= np.pi / 8:
                nms_magnitude[i, j] = 0 if magnitude[i, j] != max(magnitude[i + 1, j],
                                                                  magnitude[i, j],
                                                                  magnitude[i - 1, j]) else magnitude[i, j]
            elif 0 <= np.abs(- np.pi / 4 - angle[i, j]) <= np.pi / 8 or \
                    0 <= np.abs(3 * np.pi / 4 - angle[i, j]) <= np.pi / 8:
                nms_magnitude[i, j] = 0 if magnitude[i, j] != max(magnitude[i - 1, j + 1],
                                                                  magnitude[i, j],
                                                                  magnitude[i + 1, j - 1]) else magnitude[i, j]
            else:
                raise RuntimeError

    # show(nms_magnitude)

    T1 = 0.1 * np.max(nms_magnitude)    # weak threshold
    T2 = 0.5 * np.max(nms_magnitude)    # strong threshold

    strong_edge = (nms_magnitude > T2)
    weak_edge = np.logical_and(nms_magnitude >= T1, nms_magnitude <= T2)

    threshold_magnitude = np.zeros_like(nms_magnitude)
    for i in range(1, nms_magnitude.shape[0] - 1):
        for j in range(1, nms_magnitude.shape[1] - 1):
            if strong_edge[i, j]:
                threshold_magnitude[i, j] = 1
                for x, y in [(i-1, j-1), (i, j-1), (i+1, j-1), (i-1, j), (i-1, j), (i-1, j+1), (i, j-1), (i, j+1)]:
                    if weak_edge[x, y]:
                        threshold_magnitude[x, y] = 1
    return threshold_magnitude


def calculate_iou(bbox, bboxes):
    """
    Calculate the Intersection over Union (IoU) between a bounding box and multiple bounding boxes.

    Args:
        bbox (list): Bounding box represented as [x1, y1, x2, y2, score, class].
        bboxes (list): List of bounding boxes, each represented as [x1, y1, x2, y2, score, class].

    Returns:
        ndarray: Array of shape (N,) containing the IoU between the input bbox and each bbox in bboxes.
    """
    x1 = np.maximum(bbox[0], [box[0] for box in bboxes])
    y1 = np.maximum(bbox[1], [box[1] for box in bboxes])
    x2 = np.minimum(bbox[2], [box[2] for box in bboxes])
    y2 = np.minimum(bbox[3], [box[3] for box in bboxes])

    intersection_widths = np.maximum(0.0, x2 - x1 + 1)
    intersection_heights = np.maximum(0.0, y2 - y1 + 1)
    intersection_areas = intersection_widths * intersection_heights

    bbox_area = (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)
    bboxes_areas = [(box[2] - box[0] + 1) * (box[3] - box[1] + 1) for box in bboxes]

    union_areas = bbox_area + np.array(bboxes_areas) - intersection_areas

    ious = intersection_areas / union_areas

    return ious


def nms_multiclass(bboxes, threshold):
    """
    input: [x1, y1, x2, y2, score, class]
    """
    if len(bboxes) == 0:
        return []

    # Sort boxes by their scores in descending order
    bboxes = sorted(bboxes, key=lambda x: (x[6], x[4]), reverse=True)

    selected_bboxes = []

    while len(bboxes) > 0:
        # Get the box with the highest score
        selected_bbox = bboxes[0]

        # Add the selected box to the list
        selected_bboxes.append(selected_bbox)

        # # Filter out boxes with the same class
        # bboxes = [bbox for bbox in bboxes if bbox[5] != selected_bbox[5]]

        # Calculate the intersection over union (IoU)
        ious = calculate_iou(selected_bbox, bboxes)

        # Find the indices of bounding boxes to be suppressed
        suppressed_indices = np.where(ious > threshold)[0]

        # Remove the suppressed boxes
        bboxes = [bboxes[i] for i in range(len(bboxes)) if i not in suppressed_indices]

    return selected_bboxes


def plot_bboxes(image, bboxes):
    """
    Plot bounding boxes on the original image.

    Args:
        image (ndarray): Numpy array representing the original image.
        bboxes (list): List of bounding boxes, each represented as [x1, y1, x2, y2, score, class].
    """
    # Create a figure and axes
    matplotlib.rc("font", family='YouYuan')
    fig, ax = plt.subplots(1, dpi=300)

    # Display the image
    ax.imshow(image, cmap='gray')

    for bbox in bboxes:
        x1, y1, x2, y2, _, class_label, _ = bbox
        class_label = class_label.replace('r_cap', 'R').replace('t_cap', 'T').replace('y_cap', 'Y')

        # Create a rectangle patch
        rect = patches.Rectangle((y1, x1), y2 - y1, x2 - x1, linewidth=1, edgecolor='r', facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

        # Calculate label position
        label_x = x1
        label_y = y1 - 10

        # Ensure label does not overlap with the bounding box
        if label_y < 10:
            label_y = y1 + 10

        # Add label to the bounding box
        ax.text(label_y, label_x, class_label, color='white', fontsize=8,
                bbox=dict(facecolor='r', alpha=0.7, pad=0.2))

    plt.axis('off')
    plt.savefig(f'../data/temp/english_recognize.png')
    plt.close()


def plot_xy_histogram():
    image = cv2.imread('../data/temp/line0.png')
    # Convert the image to grayscale
    gray_image = 255 - cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the summation along the x-axis and y-axis
    x_sum = np.sum(gray_image, axis=0)
    y_sum = np.sum(gray_image, axis=1)

    # Plot the histograms
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle('Histogram Summation', fontsize=14)

    # Plot the summation along the x-axis
    ax1.plot(x_sum, color='black')
    ax1.fill_between(range(len(x_sum)), x_sum, color='black')
    ax1.set_xlabel('Y-axis')
    ax1.set_ylabel('Summation')

    # Plot the summation along the y-axis
    ax2.plot(y_sum, color='black')
    ax2.fill_between(range(len(y_sum)), y_sum, color='black')
    ax2.set_xlabel('X-axis')
    ax2.set_ylabel('Summation')

    # Adjust the spacing between the subplots
    fig.tight_layout()

    # Save the histogram plot
    plt.savefig('../data/temp/xy_histogram.png')

    # Display the histogram plot
    plt.show()

