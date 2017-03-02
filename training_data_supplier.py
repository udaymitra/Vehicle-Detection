import glob
from image_utils import *
from featurizer import get_feature_vector
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
import time

def get_train_test_data(spatial=32, histbins=32, orientations=8, pixels_per_cell=4,
                        cells_per_block=2, transform_sqrt=True):
    start_time = time.time()

    vehicle_paths = glob.glob('../vehicle_detection_data/vehicles/*/*.png')
    non_vehicle_paths = glob.glob('../vehicle_detection_data/non-vehicles/*/*.png')

    car_features = []
    notcar_features = []

    for vehicle_path in vehicle_paths:
        img = read_image_as_rgb(vehicle_path)
        car_features.append(get_feature_vector(img, spatial, histbins, orientations, pixels_per_cell, cells_per_block, transform_sqrt))

        flipped_image = flip_image(img)
        car_features.append(get_feature_vector(flipped_image, spatial, histbins, orientations, pixels_per_cell, cells_per_block, transform_sqrt))

    for vehicle_path in non_vehicle_paths:
        img = read_image_as_rgb(vehicle_path)
        notcar_features.append(get_feature_vector(img, spatial, histbins, orientations, pixels_per_cell, cells_per_block, transform_sqrt))

        flipped_image = flip_image(img)
        notcar_features.append(get_feature_vector(flipped_image, spatial, histbins, orientations, pixels_per_cell, cells_per_block, transform_sqrt))

    print("Num examples for cars: ", len(car_features))
    print("Num examples for non cars: ", len(notcar_features))

    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    y = np.hstack((np.ones(len(car_features)),
                   np.zeros(len(notcar_features))))

    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.1, random_state=rand_state)

    print("Num training examples: ", len(X_train))
    print("Num test examples: ", len(X_test))
    print("Num features: ", X_test[0].shape[0])

    elapsed_time = time.time() - start_time
    print("Time elapsed in creating training/test datasets: %.2f seconds"%elapsed_time)

    return (X_train, X_test, y_train, y_test)