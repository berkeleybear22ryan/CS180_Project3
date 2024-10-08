# THIS FILE IS FOR THE AUTOMATED FILE ... defining correspondences ...
# run this to get all the points the best way I have right now b/c i have a tone of images ... 94 i think
import os
import cv2
import numpy as np
from scipy.spatial import Delaunay
import mediapipe as mp


def get_facial_landmarks(image):
    # ... init MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True, min_detection_confidence=0.0) as face_mesh:
        # now ... process the image --> should be in RGB format
        results = face_mesh.process(image)

    if not results.multi_face_landmarks:
        print("No faces found in the image.")
        return None

    # ... use the first face found
    face_landmarks = results.multi_face_landmarks[0]

    # ... extract landmark points
    img_height, img_width, _ = image.shape
    landmark_points = []
    for lm in face_landmarks.landmark:
        x, y = int(lm.x * img_width), int(lm.y * img_height)
        landmark_points.append([x, y])

    # now just ... convert to numpy array
    points = np.array(landmark_points)
    return points




def add_boundary_points(image, points, divs=20):
    # ... get image dimensions
    h, w, _ = image.shape

    # add more boundary points and midpoints of edges to ensure coverage
    boundary_points = []
    # TODO: ... important ... get corners
    boundary_points.extend([
        [0, 0],           # top-left corner
        [w - 1, 0],       # top-right corner
        [w - 1, h - 1],   # bottom-right corner
        [0, h - 1]        # bottom-left corner
    ])


    # TODO: not sure if this helps but ... edges (divide edges into more segments for smoother results)
    divisions = divs  # increase for more points
    for i in range(1, divisions):
        boundary_points.append([i * w // divisions, 0])          # top edge
        boundary_points.append([i * w // divisions, h - 1])      # bottom edge
        boundary_points.append([0, i * h // divisions])          # left edge
        boundary_points.append([w - 1, i * h // divisions])      # right edge

    # combine with existing points
    all_points = np.vstack([points, boundary_points])
    return all_points


def draw_delaunay(image, points, triangles, color=(0, 0, 0)):
    image_copy = image.copy()
    for triangle in triangles:
        pt1 = tuple(points[triangle[0]].astype(int))
        pt2 = tuple(points[triangle[1]].astype(int))
        pt3 = tuple(points[triangle[2]].astype(int))
        cv2.line(image_copy, pt1, pt2, color, 1)
        cv2.line(image_copy, pt2, pt3, color, 1)
        cv2.line(image_copy, pt3, pt1, color, 1)
    return image_copy






if __name__ == '__main__':



    images_dir = './images_2/'
    points_dir = './points/'
    # output_dir = './images_4/'
    images5_dir = './images_5/'

    # ... ensure directories exist
    if not os.path.exists(points_dir):
        os.makedirs(points_dir)
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)

    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    image_files.sort(reverse=True)

    # TODO ... messed up the layer export number ... too long to fix -- if time come back
    print("Sorted Image Files in reverse order:")
    for img in image_files:
        # print(img)
        pass

    for i in range(len(image_files) - 1):
        image_A_filename = image_files[i]
        image_B_filename = image_files[i + 1]

        print(f"\nUsing {image_A_filename} as Image A")
        print(f"Using {image_B_filename} as Image B")

        image_A_path = os.path.join(images_dir, image_A_filename)
        image_B_path = os.path.join(images_dir, image_B_filename)

        # load images
        image_A_bgr = cv2.imread(image_A_path)
        image_B_bgr = cv2.imread(image_B_path)

        # CONFIRM ... convert BGR to RGB
        image_A = cv2.cvtColor(image_A_bgr, cv2.COLOR_BGR2RGB)
        image_B = cv2.cvtColor(image_B_bgr, cv2.COLOR_BGR2RGB)

        print(f"image_A.shape: {image_A.shape}")
        print(f"image_B.shape: {image_B.shape}")

        # use library ... get facial landmarks
        points_A = get_facial_landmarks(image_A)
        points_B = get_facial_landmarks(image_B)



        if points_A is None or points_B is None:
            print("Could not find facial landmarks in one of the images.")
            exit(2)  # ... exit code 2 mapping
            # continue

        # add boundary points ... so that we get the entire image back
        points_A = add_boundary_points(image_A, points_A, divs=20)
        points_B = add_boundary_points(image_B, points_B, divs=20)

        # CONFIRM ... both have the same number of points
        if points_A.shape[0] != points_B.shape[0]:
            print("Mismatch in the number of landmarks between images.")
            exit(3) # ... exit code 3 mapping
            # continue

        # TODO: save points to files ... might want to shorten name if time
        a_points_file_name = image_A_filename + "__A__points.txt"
        b_points_file_name = image_B_filename + "__B__points.txt"
        np.savetxt(os.path.join(points_dir, a_points_file_name), points_A)
        np.savetxt(os.path.join(points_dir, b_points_file_name), points_B)

        print("\nPoints saved successfully.")
        print(f"Number of points: {points_A.shape[0]}")












        # Compute average shape
        # so that ... Ensures that the triangle connectivity (which points form each triangle) i.e. pintis the same for both images.
        # i.e. But the best approach would probably be to compute the triangulation at a midway shape (i.e. mean of the two point sets) to lessen the potential triangle deformations.
        points_avg = (points_A + points_B) / 2.0

        # ... delaunay triangulation on the average shape
        tri = Delaunay(points_avg)

        # create the composite image
        # 1. Image A overlaid with triangulation
        image_A_triangulated = draw_delaunay(image_A, points_A, tri.simplices)

        # 2. Image B overlaid with triangulation
        image_B_triangulated = draw_delaunay(image_B, points_B, tri.simplices)

        # 3. Triangulation only for image A (black background)
        h, w, _ = image_A.shape
        triangulation_image_A = np.full_like(image_A, 0)  # black background
        triangulation_image_A = draw_delaunay(triangulation_image_A, points_A, tri.simplices, color=(255, 255, 255))

        # 4. Triangulation only for image B (black background)
        triangulation_image_B = np.full_like(image_B, 0)  # black background
        triangulation_image_B = draw_delaunay(triangulation_image_B, points_B, tri.simplices, color=(255, 255, 255))

        # Assemble the images into one big image
        # Dimensions: width = w * 2, height = h * 2
        composite_image = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)

        # ... place the images
        composite_image[0:h, 0:w] = image_A_triangulated  # top-left
        composite_image[0:h, w:2 * w] = image_B_triangulated  # top-right
        composite_image[h:2 * h, 0:w] = triangulation_image_A  # bottom-left
        composite_image[h:2 * h, w:2 * w] = triangulation_image_B  # bottom-right

        # now just ... convert composite_image from RGB to BGR for saving with cv2.imwrite
        composite_image_bgr = cv2.cvtColor(composite_image, cv2.COLOR_RGB2BGR)

        # and ... save the composite image
        composite_image_filename = f"composite_{i}_{i + 1}.jpg"
        composite_image_path = os.path.join(images5_dir, composite_image_filename)
        cv2.imwrite(composite_image_path, composite_image_bgr)

        print(f"Composite image saved to {composite_image_path}")













