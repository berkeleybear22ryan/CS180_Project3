import os
import cv2
import numpy as np
from scipy.spatial import Delaunay

from a_start3 import warp_image
from a_start4 import (
    draw_landmarks,
    read_asf_file,
    compute_average_face_shape,
    process_images_and_labels
)

if __name__ == "__main__":
    # exit(-1)

    lst_v = list(np.arange(-1, 2.1, 0.05))
    alp_0 = int(list(np.arange(-1, 2.1, 0.05))[0])
    for i in range(len(lst_v)):
        images_dir = './part2_immface/imm_face_db/'
        my_image_path = './part2_immface/immface_imports/sample_person.jpg'
        my_asf_path = './part2_immface/immface_imports/sample_person.asf'

        output_dir = './part3_immface_caricature/'

        output_dir_1 = './TRASH/immface_exports/1/'
        output_dir_2 = './TRASH/immface_exports/2/'
        all_points, all_connections, image_shapes, image_files = process_images_and_labels(images_dir, output_dir_1, output_dir_2)

        if len(all_points) == 0:
            print("No valid images found. Exiting.")
            exit()

        avg_points = compute_average_face_shape(all_points)

        user_image = cv2.imread(my_image_path)
        if user_image is None:
            print(f"Could not read user image {my_image_path}. Exiting.")
            exit()

        img_height, img_width = user_image.shape[:2]
        user_points, user_connections = read_asf_file(my_asf_path, img_width, img_height)

        # TODO: change me, just variabled it
        alpha = lst_v[i]
        delta = user_points - avg_points
        S_caricature = avg_points + alpha * delta

        def morph_face(image, source_points, destination_points):
            img_height, img_width = image.shape[:2]
            corners = np.array([
                [0, 0],
                [0, img_height - 1],
                [img_width - 1, 0],
                [img_width - 1, img_height - 1]
            ])
            source_points_extended = np.vstack([source_points, corners])
            destination_points_extended = np.vstack([destination_points, corners])
            delaunay = Delaunay(destination_points_extended)
            triangles_indices = delaunay.simplices
            morphed_img = warp_image(image, source_points_extended, destination_points_extended, triangles_indices, image.shape)
            return morphed_img


        caricature_image = morph_face(user_image, user_points, S_caricature)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        original_with_annotations = draw_landmarks(user_image, user_points, user_connections)
        # cv2.imwrite(os.path.join(output_dir, "user_original_with_annotations.jpg"), original_with_annotations)
        print("Saved user_original_with_annotations.jpg")

        caricature_with_annotations = draw_landmarks(caricature_image, S_caricature, user_connections)

        text = f"alpha: {round(alpha,4)}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        location = (50, 50)
        font_scale = 1
        color = (255, 0, 0)
        thickness = 2

        cv2.putText(caricature_image, text, location, font, font_scale, color, thickness)
        cv2.putText(caricature_with_annotations, text, location, font, font_scale, color, thickness)

        # cv2.imwrite(os.path.join(output_dir, f"F{i}__user_caricature_with_annotations.jpg"), caricature_with_annotations)
        cv2.imwrite(os.path.join(output_dir, f"{i}.jpeg"), caricature_image)
        print(f"{i/len(lst_v)}__Saved user_caricature_with_annotations.jpg")

        def draw_overlay_landmarks(image, points_list, connections, colors):
            overlay = image.copy()
            for points, color in zip(points_list, colors):
                for (start, end) in connections:
                    pt1 = tuple(np.round(points[start]).astype(int))
                    pt2 = tuple(np.round(points[end]).astype(int))
                    cv2.line(overlay, pt1, pt2, color, 1, cv2.LINE_AA)
            return overlay

        overlay_image = draw_overlay_landmarks(
            user_image,
            [user_points, S_caricature],
            user_connections,
            [(0, 255, 0), (0, 0, 255)]
        )
        # cv2.imwrite(os.path.join(output_dir, "user_annotations_overlay.jpg"), overlay_image)
        print("Saved user_annotations_overlay.jpg")
        print("Processing complete.")
