import os
import cv2
import numpy as np
from scipy.spatial import Delaunay
from a_start3 import computeAffine, bilinear_interpolate, warp_image, is_point_in_triangle

def read_asf_file(asf_path, img_width, img_height):
    points = []
    connections = []
    with open(asf_path, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith('#')]
    n_points = int(lines[0])
    for i in range(1, n_points + 1):
        tokens = lines[i].split()
        if len(tokens) < 7:
            continue
        path_num = tokens[0]
        type = tokens[1]
        x_rel = tokens[2]
        y_rel = tokens[3]
        point_num = int(tokens[4])
        conn_from = int(tokens[5])
        conn_to = int(tokens[6])
        x = float(x_rel) * img_width
        y = float(y_rel) * img_height
        points.append([x, y])
        if conn_from != point_num:
            connections.append((point_num, conn_from))
        if conn_to != point_num:
            connections.append((point_num, conn_to))
    points = np.array(points)
    return points, connections

def draw_landmarks(image, points, connections, show_points=True, show_lines=True):
    img_copy = image.copy()
    if show_lines:
        for (start, end) in connections:
            if start != end and start < len(points) and end < len(points):
                start_point = tuple(points[start].astype(int))
                end_point = tuple(points[end].astype(int))
                cv2.line(img_copy, start_point, end_point, (255, 0, 0), 1)
    if show_points:
        for (x, y) in points.astype(int):
            cv2.circle(img_copy, (x, y), 2, (0, 255, 0), -1)
    return img_copy

def process_images_and_labels(images_dir, output_dir_1, output_dir_2):
    if not os.path.exists(output_dir_1):
        os.makedirs(output_dir_1)
    if not os.path.exists(output_dir_2):
        os.makedirs(output_dir_2)
    all_points = []
    all_connections = []
    image_shapes = []
    image_files = []
    for filename in sorted(os.listdir(images_dir)):
        if filename.lower().endswith(('.jpg', '.bmp', '.png')):
            image_path = os.path.join(images_dir, filename)
            base_name = os.path.splitext(filename)[0]
            asf_path = os.path.join(images_dir, base_name + '.asf')
            if not os.path.exists(asf_path):
                print(f"ASF file not found for {filename}. Skipping.")
                continue
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not read image {filename}. Skipping.")
                continue
            img_height, img_width = image.shape[:2]
            try:
                points, connections = read_asf_file(asf_path, img_width, img_height)
                # ... (1) points and annotations overlaid with images
                img_with_landmarks = draw_landmarks(image, points, connections)
                cv2.imwrite(os.path.join(output_dir_1, filename), img_with_landmarks)
                # ... (2) points and annotations NOT overlaid with images (on a blank image)
                blank_image = np.zeros_like(image)
                landmarks_on_blank = draw_landmarks(blank_image, points, connections)
                cv2.imwrite(os.path.join(output_dir_2, filename), landmarks_on_blank)
                all_points.append(points)
                all_connections.append(connections)
                image_shapes.append((img_height, img_width))
                image_files.append(image_path)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    return all_points, all_connections, image_shapes, image_files

def overlay_annotations_incrementally(all_points, all_connections, image_shape, output_dir_3):
    if not os.path.exists(output_dir_3):
        os.makedirs(output_dir_3)
    accumulated_image = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)
    for idx, (points, connections) in enumerate(zip(all_points, all_connections)):
        accumulated_image = draw_landmarks(accumulated_image, points, connections)
        cv2.imwrite(os.path.join(output_dir_3, f"overlay_{idx+1}.jpg"), accumulated_image)

def compute_average_shape_incrementally(all_points, all_connections, image_shape, output_dir_4):
    if not os.path.exists(output_dir_4):
        os.makedirs(output_dir_4)
    accumulated_points = []
    for idx, points in enumerate(all_points):
        accumulated_points.append(points)
        avg_points = np.mean(np.array(accumulated_points), axis=0)
        avg_image = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)
        # ... will use connections from the first image
        first_connections = all_connections[0]
        avg_image = draw_landmarks(avg_image, avg_points, first_connections)
        cv2.putText(avg_image, f"Average: {idx+1}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.imwrite(os.path.join(output_dir_4, f"average_{idx+1}.jpg"), avg_image)

def compute_average_face_shape(all_points):
    mean_shape = np.mean(np.array(all_points), axis=0)
    return mean_shape

def plot_average_face_shape(mean_shape, connections, image_shape, output_dir_5):
    if not os.path.exists(output_dir_5):
        os.makedirs(output_dir_5)
    annotated_image = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)
    annotated_image = draw_landmarks(annotated_image, mean_shape, connections)
    cv2.imwrite(os.path.join(output_dir_5, "average_shape_annotated.jpg"), annotated_image)

    average_face_path = os.path.join(output_dir_5, "average_face.jpg")
    if os.path.exists(average_face_path):
        average_face = cv2.imread(average_face_path)
        average_face_annotated = draw_landmarks(average_face, mean_shape, connections)
        cv2.imwrite(os.path.join(output_dir_5, "average_faceandshape_annotated.jpg"), average_face_annotated)
    else:
        print("Average face image not found when attempting to create average_faceandshape_annotated.jpg")
    unannotated_image = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)
    cv2.imwrite(os.path.join(output_dir_5, "average_shape_unannotated.jpg"), unannotated_image)

def morph_face_to_average_custom(image, points, avg_points):
    try:
        img_height, img_width = image.shape[:2]
        # REALLY IMPORTANT ... add image corners to points and avg_points
        corners = np.array([[0, 0], [0, img_height - 1], [img_width - 1, 0], [img_width - 1, img_height - 1]])
        points_extended = np.vstack([points, corners])
        avg_points_extended = np.vstack([avg_points, corners])

        # # remove duplicate points (if any) -- for some reason was breaking big time ... check why if time
        # avg_points_extended = np.unique(avg_points_extended, axis=0)
        # points_extended = np.unique(points_extended, axis=0)

        # ... delaunay triangulation on the average shape
        delaunay = Delaunay(avg_points_extended)
        triangles_indices = delaunay.simplices

        dest_shape = image.shape
        morphed_img = warp_image(image, points_extended, avg_points_extended, triangles_indices, dest_shape)
        return morphed_img
    except Exception as e:
        print(f"Exception in morph_face_to_average_custom: {e}")
        return None

def morph_faces_to_average_custom(all_points, all_connections, all_images, avg_points, avg_connections, output_dir_6):
    if not os.path.exists(output_dir_6):
        os.makedirs(output_dir_6)
    for idx, (image_path, points, connections) in enumerate(zip(all_images, all_points, all_connections)):
        image = cv2.imread(image_path)
        morphed_image = morph_face_to_average_custom(image, points, avg_points)
        # ... annotate original image
        image_annotated = draw_landmarks(image, points, connections)
        # ... annotate morphed image
        morphed_image_annotated = draw_landmarks(morphed_image, avg_points, avg_connections)
        # ... combine images side by side
        combined_image = np.hstack((image_annotated, morphed_image_annotated))
        cv2.imwrite(os.path.join(output_dir_6, f"morphed_{idx+1}_annotated.jpg"), combined_image)
        # ... also save without annotations if needed
        combined_image_unannotated = np.hstack((image, morphed_image))
        cv2.imwrite(os.path.join(output_dir_6, f"morphed_{idx+1}.jpg"), combined_image_unannotated)

def compute_average_face_image_custom(all_points, all_images, avg_points, output_dir_5):
    print("Starting computation of average face image.")
    avg_face = None
    count = 0  # To keep track of valid images
    for idx, (image_path, points) in enumerate(zip(all_images, all_points)):
        print(f"Processing image {idx+1}/{len(all_images)}: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read image {image_path}. Skipping.")
            continue
        try:
            morphed_image = morph_face_to_average_custom(image, points, avg_points)
            if morphed_image is not None and morphed_image.size != 0:
                if avg_face is None:
                    avg_face = morphed_image.astype(np.float32)
                else:
                    avg_face += morphed_image.astype(np.float32)
                count += 1
                print(f"Successfully processed and added image {idx+1} to average.")
            else:
                print(f"Failed to morph image {idx+1}.")
        except Exception as e:
            print(f"Exception while processing image {idx+1}: {e}")
    if count > 0:
        avg_face /= count
        avg_face = avg_face.astype(np.uint8)
        if not os.path.exists(output_dir_5):
            os.makedirs(output_dir_5)
        cv2.imwrite(os.path.join(output_dir_5, "average_face.jpg"), avg_face)
        print(f"Saved average_face.jpg after processing {count} images.")
    else:
        print("No valid images to compute the average face.")

def process_user_face_custom(user_image_path, user_asf_path, avg_points, avg_connections, output_dir_7, output_dir_8, output_dir_5):
    if not os.path.exists(output_dir_7):
        os.makedirs(output_dir_7)
    if not os.path.exists(output_dir_8):
        os.makedirs(output_dir_8)
    user_image = cv2.imread(user_image_path)
    if user_image is None:
        print(f"Could not read user image {user_image_path}.")
        return
    img_height, img_width = user_image.shape[:2]
    user_points, user_connections = read_asf_file(user_asf_path, img_width, img_height)
    # ... (1) user face warped into average geometry
    user_face_in_avg_geom = morph_face_to_average_custom(user_image, user_points, avg_points)
    cv2.imwrite(os.path.join(output_dir_7, "user_face_in_avg_geometry.jpg"), user_face_in_avg_geom)
    # ... now just annotate the morphed image
    user_face_in_avg_geom_annotated = draw_landmarks(user_face_in_avg_geom, avg_points, avg_connections)
    cv2.imwrite(os.path.join(output_dir_7, "user_face_in_avg_geometry_annotated.jpg"), user_face_in_avg_geom_annotated)
    # AND ... (2) average face warped into user's geometry
    avg_face_path = os.path.join(output_dir_5, "average_face.jpg")
    if not os.path.exists(avg_face_path):
        print("Average face image not found. Exiting.")
        return
    avg_face = cv2.imread(avg_face_path)

    def morph_average_face_to_user_custom(avg_face, avg_points, user_points):
        img_height, img_width = avg_face.shape[:2]
        corners = np.array([[0, 0], [0, img_height - 1], [img_width - 1, 0], [img_width - 1, img_height - 1]])
        avg_points_extended = np.vstack([avg_points, corners])
        user_points_extended = np.vstack([user_points, corners])
        delaunay = Delaunay(user_points_extended)
        triangles_indices = delaunay.simplices
        dest_shape = avg_face.shape
        morphed_img = warp_image(avg_face, avg_points_extended, user_points_extended, triangles_indices, dest_shape)
        return morphed_img

    avg_face_in_user_geom = morph_average_face_to_user_custom(avg_face, avg_points, user_points)
    cv2.imwrite(os.path.join(output_dir_8, "avg_face_in_user_geometry.jpg"), avg_face_in_user_geom)
    avg_face_in_user_geom_annotated = draw_landmarks(avg_face_in_user_geom, user_points, user_connections)
    cv2.imwrite(os.path.join(output_dir_8, "avg_face_in_user_geometry_annotated.jpg"), avg_face_in_user_geom_annotated)
    user_annotated = draw_landmarks(user_image, user_points, user_connections)
    cv2.imwrite(os.path.join(output_dir_7, "user_face_annotated.jpg"), user_annotated)

if __name__ == "__main__":
    images_dir = './part2_immface/imm_face_db/'
    my_image_path = './part2_immface/immface_imports/sample_person.jpg'
    my_asf_path = './part2_immface/immface_imports/sample_person.asf'

    output_dir_1 = './part2_immface/immface_exports/1/'
    output_dir_2 = './part2_immface/immface_exports/2/'
    output_dir_3 = './part2_immface/immface_exports/3/'
    output_dir_4 = './part2_immface/immface_exports/4/'
    output_dir_5 = './part2_immface/immface_exports/5/'
    output_dir_6 = './part2_immface/immface_exports/6/'
    output_dir_7 = './part2_immface/immface_exports/7/'
    output_dir_8 = './part2_immface/immface_exports/8/'

    all_points, all_connections, image_shapes, image_files = process_images_and_labels(images_dir, output_dir_1, output_dir_2)

    if len(all_points) == 0:
        print("No valid images found. Exiting.")
        exit()
    image_shape = image_shapes[0]
    overlay_annotations_incrementally(all_points, all_connections, image_shape, output_dir_3)
    compute_average_shape_incrementally(all_points, all_connections, image_shape, output_dir_4)
    avg_points = compute_average_face_shape(all_points)
    first_connections = all_connections[0]
    compute_average_face_image_custom(all_points, image_files, avg_points, output_dir_5)
    plot_average_face_shape(avg_points, first_connections, image_shape, output_dir_5)
    morph_faces_to_average_custom(all_points, all_connections, image_files, avg_points, first_connections, output_dir_6)
    process_user_face_custom(my_image_path, my_asf_path, avg_points, first_connections, output_dir_7, output_dir_8, output_dir_5)
    print("Processing complete.")
