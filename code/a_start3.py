import os
import cv2
import numpy as np
from scipy.spatial import Delaunay


def computeAffine(src_tri_pts, dest_tri_pts):
    src_tri_pts_hom = np.hstack((src_tri_pts, np.ones((3, 1))))
    A = np.linalg.lstsq(src_tri_pts_hom, dest_tri_pts, rcond=None)[0].T
    return A


def is_point_in_triangle(points, triangle):
    A = triangle[0]
    B = triangle[1]
    C = triangle[2]
    v0 = C - A
    v1 = B - A
    v2 = points - A

    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2.T)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2.T)

    denom = dot00 * dot11 - dot01 * dot01

    if np.abs(denom) < 1e-6:
        return np.zeros(len(points), dtype=bool)

    invDenom = 1 / denom
    u = (dot11 * dot02 - dot01 * dot12) * invDenom
    v = (dot00 * dot12 - dot01 * dot02) * invDenom

    mask = (u >= 0) & (v >= 0) & (u + v <= 1)
    return mask


def bilinear_interpolate(img, x, y):
    x0 = np.floor(x).astype(np.int32)
    x1 = x0 + 1
    y0 = np.floor(y).astype(np.int32)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, img.shape[1] - 1)
    x1 = np.clip(x1, 0, img.shape[1] - 1)
    y0 = np.clip(y0, 0, img.shape[0] - 1)
    y1 = np.clip(y1, 0, img.shape[0] - 1)

    Ia = img[y0, x0]
    Ib = img[y1, x0]
    Ic = img[y0, x1]
    Id = img[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    interpolated = wa[:, None] * Ia + wb[:, None] * Ib + wc[:, None] * Ic + wd[:, None] * Id
    return interpolated


def warp_image(src_img, src_points, dest_points, triangles, dest_shape):
    warped_img = np.zeros(dest_shape, dtype=src_img.dtype)

    h, w = dest_shape[:2]

    # now ... for each triangle, perform the affine warp ... check this runtime ...
    for idx, tri_indices in enumerate(triangles):
        # ... getting the coordinates of the triangle vertices in source and destination images
        # confirm ...
        src_tri = src_points[tri_indices]
        dest_tri = dest_points[tri_indices]

        # then we want to compute bounding rectangle for the destination triangle
        x_min = int(np.floor(np.min(dest_tri[:, 0])))
        x_max = int(np.ceil(np.max(dest_tri[:, 0])))
        y_min = int(np.floor(np.min(dest_tri[:, 1])))
        y_max = int(np.ceil(np.max(dest_tri[:, 1])))

        # and ... clip the rectangle to boundaries of the image
        x_min = max(x_min, 0)
        x_max = min(x_max, w - 1)
        y_min = max(y_min, 0)
        y_max = min(y_max, h - 1)

        # then will create a meshgrid of pixel coordinates that are within the bounding rectangle ...
        # check ...
        X, Y = np.meshgrid(np.arange(x_min, x_max + 1), np.arange(y_min, y_max + 1))
        coords = np.vstack((X.flatten(), Y.flatten())).T
        # print(coords)

        # now make sure to check which points are inside the destination triangle ...
        mask = is_point_in_triangle(coords, dest_tri)

        coords_in_tri = coords[mask]

        if coords_in_tri.shape[0] == 0:
            continue

        # now ... compute the affine transformation matrix A
        A = computeAffine(dest_tri, src_tri)

        # ... convert to homogeneous coordinates by adding a column of ones
        ones = np.ones((coords_in_tri.shape[0], 1))
        coords_in_tri_hom = np.hstack((coords_in_tri, ones))

        # next make sure to apply inverse mapping and will get corresponding coordinates in the source image
        src_coords = A @ coords_in_tri_hom.T

        src_x = src_coords[0, :]
        src_y = src_coords[1, :]

        interpolated_pixels = bilinear_interpolate(src_img, src_x, src_y)
        warped_img[coords_in_tri[:, 1], coords_in_tri[:, 0]] = interpolated_pixels

    return warped_img


def morph(im1, im2, im1_pts, im2_pts, triangles, warp_frac, dissolve_frac):
    intermediate_pts = (1 - warp_frac) * im1_pts + warp_frac * im2_pts
    h, w, _ = im1.shape
    warp_shape = (h, w, 3)
    warped_im1 = warp_image(im1, im1_pts, intermediate_pts, triangles, warp_shape)
    warped_im2 = warp_image(im2, im2_pts, intermediate_pts, triangles, warp_shape)
    morphed_im = (1 - dissolve_frac) * warped_im1 + dissolve_frac * warped_im2
    morphed_im = morphed_im.astype(np.uint8)
    return morphed_im



# classic sharpen ... to make the image look better, idea was might help with low freq during transition period ...
def sharpen_image(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]], dtype=np.float32)

    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened


def draw_triangulation(img, points, triangles, color=(0, 255, 0)):
    for tri in triangles:
        pts = points[tri].astype(np.int32)
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=1)
    return img


if __name__ == '__main__':

    # important directories ...
    images_dir = './images_2/'
    points_dir = './points/'
    morph_frames_dir = './morph_frames/'
    morph_frames_sharpened_dir = './morph_frames_sharpened/'
    morph_frames_sharpened_composite_dir = './morph_frames_sharpened_composite/'
    mid_way_faces_dir = './mid_way_faces/'

    os.makedirs(points_dir, exist_ok=True)
    os.makedirs(morph_frames_dir, exist_ok=True)
    os.makedirs(morph_frames_sharpened_dir, exist_ok=True)
    os.makedirs(morph_frames_sharpened_composite_dir, exist_ok=True)
    os.makedirs(mid_way_faces_dir, exist_ok=True)

    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    image_files.sort(reverse=True)

    for i in range(len(image_files) - 1):
        image_A_filename = image_files[i]
        image_B_filename = image_files[i + 1]

        print(f"\nUsing {image_A_filename} as Image A")
        print(f"Using {image_B_filename} as Image B")

        image_A_path = os.path.join(images_dir, image_A_filename)
        image_B_path = os.path.join(images_dir, image_B_filename)

        # ... load images
        image_A_bgr = cv2.imread(image_A_path)
        image_B_bgr = cv2.imread(image_B_path)

        # ... convert BGR to RGB
        image_A = cv2.cvtColor(image_A_bgr, cv2.COLOR_BGR2RGB)
        image_B = cv2.cvtColor(image_B_bgr, cv2.COLOR_BGR2RGB)

        print(f"image_A.shape: {image_A.shape}")
        print(f"image_B.shape: {image_B.shape}")

        # ... ensure images are the same size
        if image_A.shape != image_B.shape:
            print("Images have different sizes. Resizing image B to match image A.")
            image_B = cv2.resize(image_B, (image_A.shape[1], image_A.shape[0]))

        # TODO: to save recomputing ... just load points from files assuming there were generated earlier
        a_points_file_name = image_A_filename + "__A__points.txt"
        b_points_file_name = image_B_filename + "__B__points.txt"

        points_A_path = os.path.join(points_dir, a_points_file_name)
        points_B_path = os.path.join(points_dir, b_points_file_name)

        if not os.path.exists(points_A_path) or not os.path.exists(points_B_path):
            print("Point files not found for one of the images.")
            continue

        points_A = np.loadtxt(points_A_path).astype(np.float32)
        points_B = np.loadtxt(points_B_path).astype(np.float32)

        # TODO: ... important, ensure both have the same number of points
        if points_A.shape != points_B.shape:
            print("Mismatch in the number of landmarks between images.")
            continue

        print(f"Number of points: {points_A.shape[0]}")

        # ... getting the average shape
        points_avg = (points_A + points_B) / 2.0

        # ... Delaunay triangulation on the average shape
        tri = Delaunay(points_avg)
        triangles = tri.simplices

        h, w, _ = image_A.shape
        warp_shape = (h, w, 3)

        warped_im1 = warp_image(image_A, points_A, points_avg, triangles, warp_shape)
        warped_im2 = warp_image(image_B, points_B, points_avg, triangles, warp_shape)

        mid_way_face = (warped_im1.astype(np.float32) + warped_im2.astype(np.float32)) / 2.0
        mid_way_face = mid_way_face.astype(np.uint8)

        mid_way_face_bgr = cv2.cvtColor(mid_way_face, cv2.COLOR_RGB2BGR)
        mid_way_face_filename = f"mid_way_face_{i}_{i + 1}.jpg"
        cv2.imwrite(os.path.join(mid_way_faces_dir, mid_way_face_filename), mid_way_face_bgr)

        print(f"Mid-way face saved to {mid_way_faces_dir}/{mid_way_face_filename}")

        num_frames = 45

        for frame in range(num_frames + 1):
            warp_frac = frame / num_frames
            dissolve_frac = warp_frac

            morphed_frame = morph(image_A, image_B, points_A, points_B, triangles, warp_frac, dissolve_frac)

            # extra ... sharpen the morphed frame
            sharpened_frame = sharpen_image(morphed_frame)

            # ... save the original morphed frame
            frame_bgr = cv2.cvtColor(morphed_frame, cv2.COLOR_RGB2BGR)
            frame_filename = f"morph_{i}_{i + 1}_frame_{frame:03d}.jpg"
            cv2.imwrite(os.path.join(morph_frames_dir, frame_filename), frame_bgr)

            # ... save the sharpened frame
            sharpened_frame_bgr = cv2.cvtColor(sharpened_frame, cv2.COLOR_RGB2BGR)
            sharpened_frame_filename = f"sharpened_morph_{i}_{i + 1}_frame_{frame:03d}.jpg"
            cv2.imwrite(os.path.join(morph_frames_sharpened_dir, sharpened_frame_filename), sharpened_frame_bgr)

            # creating composite image (3x2 grid) ...
            composite_image = np.zeros((h * 2, w * 3, 3), dtype=np.uint8)

            image_A_resized = cv2.resize(image_A, (w, h))
            image_B_resized = cv2.resize(image_B, (w, h))

            image_A_with_tris = draw_triangulation(image_A_resized.copy(), points_A, triangles)
            image_B_with_tris = draw_triangulation(image_B_resized.copy(), points_B, triangles)

            intermediate_pts = (1 - warp_frac) * points_A + warp_frac * points_B

            morphed_frame_with_tris = draw_triangulation(sharpened_frame.copy(), intermediate_pts, triangles)

            black_img = np.zeros((h, w, 3), dtype=np.uint8)

            image_A_tris_only = draw_triangulation(black_img.copy(), points_A, triangles)
            morphed_tris_only = draw_triangulation(black_img.copy(), intermediate_pts, triangles)  # ... update here
            image_B_tris_only = draw_triangulation(black_img.copy(), points_B, triangles)

            composite_image[0:h, 0:w] = image_A_with_tris  # ... top-left
            composite_image[0:h, w:2 * w] = morphed_frame_with_tris  # ... top-middle
            composite_image[0:h, 2 * w:3 * w] = image_B_with_tris  # ... top-right
            composite_image[h:2 * h, 0:w] = image_A_tris_only  # ... bottom-left
            composite_image[h:2 * h, w:2 * w] = morphed_tris_only  # ... bottom-middle, updated to use intermediate_pts
            composite_image[h:2 * h, 2 * w:3 * w] = image_B_tris_only  # ... bottom-right

            # make sure to convert composite image to BGR for saving
            composite_image_bgr = cv2.cvtColor(composite_image, cv2.COLOR_RGB2BGR)

            # now save the composite image
            composite_filename = f"composite_{i}_{i + 1}_frame_{frame:03d}.jpg"
            cv2.imwrite(os.path.join(morph_frames_sharpened_composite_dir, composite_filename), composite_image_bgr)

        print(f"Morph sequence frames saved to {morph_frames_dir}")
        print(f"Sharpened frames saved to {morph_frames_sharpened_dir}")
        print(f"Composite images saved to {morph_frames_sharpened_composite_dir}")
