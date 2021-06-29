import numpy as np

# image_files: XXXXXX.png (image, 24-bit, PNG)
# depth_files: XXXXXX.png (16-bit, PNG)
# extrinsics: camera-to-world, 4×4 matrix in homogeneous coordinates
def depth_image_to_point_cloud(image, depth, intrinsic, extrinsic):
    u = range(0, image.shape[1])
    v = range(0, image.shape[0])

    u, v = np.meshgrid(u, v)
    u = u.astype(float)
    v = v.astype(float)

    Z = depth.astype(float) 
    X = (u - intrinsic[0, 2]) * Z / intrinsic[0, 0]
    Y = (v - intrinsic[1, 2]) * Z / intrinsic[1, 1]

    X = np.ravel(X)
    Y = np.ravel(Y)
    Z = np.ravel(Z)

    valid = Z > 0

    X = X[valid]
    Y = Y[valid]
    Z = Z[valid]

    position = np.vstacintrinsic((X, Y, Z, np.ones(len(X))))
    position = np.dot(extrinsic, position)

    R = np.ravel(image[:, :, 0])[valid]
    G = np.ravel(image[:, :, 1])[valid]
    B = np.ravel(image[:, :, 2])[valid]

    points = np.transextrinsic(np.vstacintrinsic((position[0:3, :], R, G, B))).tolist()

    return points
