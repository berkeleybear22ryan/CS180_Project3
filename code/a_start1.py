# THIS FILE IS FOR THE MANUAL ... defining correspondences ... but found a way that is better and allows me more images ... so can ignore
# I left this here because they say they wanted it ...
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay


# TODO: IMPORTANT ... you must alternate selecting points o/w the Delaunay down the road will fail
def get_corresponding_points(image_A, image_B):
    points_A = []
    points_B = []

    print("You'll be selecting points for corresponding facial features.")
    print("Press 'q' to stop selecting points.")

    # ask the user for display mode ...
    display_mode = input("Enter '1' for side-by-side display or '2' for overlay display: ")
    while display_mode not in ['1', '2']:
        display_mode = input("Invalid input. Please enter '1' or '2': ")

    # get ... opacity values if overlay mode is selected
    if display_mode == '2':
        try:
            opacity_A = float(input("Enter opacity for Image A (0.0 to 1.0, e.g., 0.5): "))
            opacity_B = float(input("Enter opacity for Image B (0.0 to 1.0, e.g., 0.5): "))
        except ValueError:
            print("Invalid input. Using default opacity values of 0.5.")
            opacity_A = 0.5
            opacity_B = 0.5

        # CHECK ... opacity values are within bounds
        opacity_A = max(0.0, min(1.0, opacity_A))
        opacity_B = max(0.0, min(1.0, opacity_B))

    # figure window for selecting points
    if display_mode == '1':
        # METHOD: Side-by-side display
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        axs[0].imshow(image_A)
        axs[0].set_title("Image A")
        axs[0].axis('off')

        axs[1].imshow(image_B)
        axs[1].set_title("Image B")
        axs[1].axis('off')

        # grid lines
        grid_spacing = 10

        for ax in axs:
            # ... image dimensions
            height, width, _ = image_A.shape  # ASSUME: both images are the same size

            x_positions = np.arange(0, width, grid_spacing)
            y_positions = np.arange(0, height, grid_spacing)

            # ... draw vertical grid lines
            for x in x_positions:
                ax.axvline(x, color='gray', linestyle='-', linewidth=0.5)

            # ... draw horizontal grid lines
            for y in y_positions:
                ax.axhline(y, color='gray', linestyle='-', linewidth=0.5)

    else:
        # ... overlay display
        blended_image = cv2.addWeighted(image_A, opacity_A, image_B, opacity_B, 0)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(blended_image)
        ax.set_title(f"Overlay Mode (Opacity A: {opacity_A}, Opacity B: {opacity_B})")
        ax.axis('off')

        # grid lines
        grid_spacing = 10

        # image dimensions
        height, width, _ = blended_image.shape

        x_positions = np.arange(0, width, grid_spacing)
        y_positions = np.arange(0, height, grid_spacing)

        # vertical grid lines
        for x in x_positions:
            ax.axvline(x, color='gray', linestyle='-', linewidth=0.5)

        # horizontal grid lines
        for y in y_positions:
            ax.axhline(y, color='gray', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    plt.show(block=False)

    print("Click points on the images. Press 'q' to finish.")

    # ... variables to manage point collection
    collecting_points = True

    # this is the event handling for continuous point selection
    def onclick(event):
        nonlocal collecting_points

        if not collecting_points:
            return

        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return  # when you click was outside the axes

        if display_mode == '1':
            # METHOD: ... side-by-side mode
            if event.inaxes == axs[0]:
                # if ... clicked on Image A
                points_A.append((x, y))
                axs[0].plot(x, y, 'ro')
                plt.draw()
                print(f"Point added to Image A: ({x:.2f}, {y:.2f})")
            elif event.inaxes == axs[1]:
                # if ... clicked on Image B
                points_B.append((x, y))
                axs[1].plot(x, y, 'bo')
                plt.draw()
                print(f"Point added to Image B: ({x:.2f}, {y:.2f})")
        else:
            # OVERLAY MODE ...
            # METHOD: use mouse button to distinguish between images
            if event.button == 1:  # ... left click for Image A
                points_A.append((x, y))
                ax.plot(x, y, 'ro')
                plt.draw()
                print(f"Point added to Image A: ({x:.2f}, {y:.2f})")
            elif event.button == 3:  # ... right click for Image B
                points_B.append((x, y))
                ax.plot(x, y, 'bo')
                plt.draw()
                print(f"Point added to Image B: ({x:.2f}, {y:.2f})")
            else:
                print("Use left click for Image A, right click for Image B.")

    def onkey(event):
        nonlocal collecting_points
        if event.key == 'q':
            collecting_points = False
            plt.close()

    if display_mode == '1':
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        kid = fig.canvas.mpl_connect('key_press_event', onkey)
    else:
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        kid = fig.canvas.mpl_connect('key_press_event', onkey)

    print("Instructions:")
    if display_mode == '1':
        print(" - Click on Image A (left) or Image B (right) to add points.")
    else:
        print(" - Left click to add a point to Image A.")
        print(" - Right click to add a point to Image B.")
    print(" - Press 'q' to stop selecting points and close the window.")

    plt.show()

    min_len = min(len(points_A), len(points_B))
    points_A = points_A[:min_len]
    points_B = points_B[:min_len]

    return np.array(points_A), np.array(points_B)

# way to manually get the points that will be used in the Delaunay
if __name__ == '__main__':
    images_dir = './images_2/'
    points_dir = './points/'
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    image_files.sort(reverse=True)
    print("Sorted Image Files in reverse order:")
    for img in image_files:
        pass
        # print(img)


    image_A_filename = image_files[0]  # EX: L__0093_MS_Hilton_Paris_CloseUp.jpg
    image_B_filename = image_files[1]  # EX: L__0092_MS_Gevinson_Tavi_CloseUp.jpg

    print(f"Using {image_A_filename} as image A")
    print(f"Using {image_B_filename} as image B")

    image_A_path = images_dir + image_A_filename
    image_B_path = images_dir + image_B_filename

    image_A = cv2.imread(image_A_path)
    image_B = cv2.imread(image_B_path)
    print(f"image_A.shape: {image_A.shape}")
    print(f"image_B.shape: {image_B.shape}")

    image_A = cv2.cvtColor(image_A, cv2.COLOR_BGR2RGB)
    image_B = cv2.cvtColor(image_B, cv2.COLOR_BGR2RGB)

    points_A, points_B = get_corresponding_points(image_A, image_B)
    a_points_file_name = image_A_filename + "__A__points.txt"
    b_points_file_name = image_B_filename + "__B__points.txt"
    np.savetxt(points_dir + a_points_file_name, points_A)
    np.savetxt(points_dir + b_points_file_name, points_B)

    print("Points saved successfully.")
    print(f"Image A points: {points_A}")
    print(f"Image B points: {points_B}")