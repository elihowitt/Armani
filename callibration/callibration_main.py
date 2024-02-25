import random
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

def get_rand_point():
    x = random.randint(1, 50)
    y = random.randint(1, 50)
    z = random.randint(1, 50)
    return [x, y, z]

def main():

    # notation:
    # for a given point in space, (x1, y1, z1) is its representation
    # in the cameras perspective, and (x2, y2, z2) in the lazers.
    # (dx, dy, dz) and (ax, ay, az) are such that
    # (x2, y2, z2) = M * (x1 + dx, y1 + dy, z1 + dz) with M the rotation matrix of angles (ax, ay, az)

    # creating "real" values for testing:
    dx = 1.3
    dy = 1.8
    dz = 3.4
    ax = 0.1
    ay = -0.05
    az = 0.05
    real_rot = Rotation.from_euler('xyz', [ax, ay, az])
    real_trans = [dx, dy, dz]

    def real_cam_to_lazer(point):
        translated = [sum(_) for _ in zip(point, real_trans)]
        return real_rot.apply(translated)

    # dataset of points in the format points[i] = [[x1_i, y1_i, z1_i], [x2_i, y2_i, z2_i]]
    # initializing dataset:
    NUM_DATAPOINTS = 1000
    datapoints = []
    for i in range(NUM_DATAPOINTS):
        campoint = get_rand_point()
        lazerpoint = real_cam_to_lazer(campoint)
        datapoints.append([campoint, lazerpoint])

    # defining the error function for guessing x=[dx, dy, dz, ax, ay, az] values
    def error_function(x):
        noise = 0.01*np.random.normal(0, 0.001, size=(NUM_DATAPOINTS, 3))
        r = Rotation.from_euler('xyz', [x[3], x[4], x[5]])
        t = [x[0], x[1], x[2]]
        error = 0
        for idx, [campoint, lazerpoint] in enumerate(datapoints):
            noisyinput = [sum(_) for _ in zip(campoint, noise[idx])]
            translated_point = [sum(_) for _ in zip(noisyinput, t)]

            guess = r.apply(translated_point)
            for j in range(3):
                error += (guess[j]-lazerpoint[j])**2
        return error

    # guess for distance and orientation values:
    guess = [1, 2, 3, 0, 0, 0]
    res = minimize(error_function, guess)
    print(res)






if __name__ == '__main__':
    main()