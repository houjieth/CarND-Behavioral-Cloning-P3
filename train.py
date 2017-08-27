import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D

from sklearn.model_selection import train_test_split


TRAINING_BATCH_SIZE = 384  # 3 * 128
VALIDATION_BATCH_SIZE = 128  # 1 * 128
EPOCH_COUNT = 5


def process_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # plt.imshow(image)
    # plt.show(block=True)
    return image


def read_driving_log(driving_log_path):
    log_lines = []
    with open(driving_log_path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            log_lines.append(line)
    return log_lines


def training_data_generator(training_log_lines, batch_size):
    images = []
    steerings = []
    data_count = 0

    while True:
        for line in training_log_lines:
            center_image_path, left_image_path, right_image_path = line[0], line[1], line[2]

            # center image
            image = cv2.imread(center_image_path)
            image = process_image(image)
            steering = float(line[3])
            images.append(image)
            steerings.append(steering)

            # left image
            image = cv2.imread(left_image_path)
            image = process_image(image)
            steering = float(line[3]) + 1
            images.append(image)
            steerings.append(steering)

            # right image
            image = cv2.imread(right_image_path)
            image = process_image(image)
            steering = float(line[3]) - 1
            images.append(image)
            steerings.append(steering)

            data_count += 3
            if data_count == batch_size:
                yield (np.array(images), np.array(steerings))
                del images[:]
                del steerings[:]
                data_count = 0

        # yield last batch if there's still data
        if data_count:
            yield (np.array(images), np.array(steerings))

        del images[:]
        del steerings[:]
        data_count = 0


def validation_data_generator(validation_log_lines, batch_size):
    images = []
    steerings = []
    data_count = 0

    while True:
        for line in validation_log_lines:
            # only use the center image because we will only use ground truth in validation data
            center_image_path = line[0]
            image = cv2.imread(center_image_path)
            image = process_image(image)
            steering = float(line[3])
            images.append(image)
            steerings.append(steering)

            data_count += 1
            if data_count == batch_size:
                yield (np.array(images), np.array(steerings))
                del images[:]
                del steerings[:]
                data_count = 0

        # yield last batch if there's still data
        if data_count:
            yield (np.array(images), np.array(steerings))

        del images[:]
        del steerings[:]
        data_count = 0


if __name__ == "__main__":
    log_lines = read_driving_log('data_1/driving_log.csv')
    training_log_lines, validation_log_lines = train_test_split(log_lines, test_size=0.2)

    training_generator = training_data_generator(training_log_lines, TRAINING_BATCH_SIZE)
    validation_generator = validation_data_generator(validation_log_lines, VALIDATION_BATCH_SIZE)

    # for batch in training_generator:
    #     pass

    # for batch in validation_generator:
    #     if batch is None:
    #         print("SHIT")

    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    print(model.output_shape)
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    print(model.output_shape)
    model.add(Flatten())
    print(model.output_shape)
    model.add(Dense(1))
    print(model.output_shape)

    model.compile(loss='mse', optimizer='adam')
    model.fit_generator(training_generator, len(training_log_lines) * 3, nb_epoch=5,
                        validation_data=validation_generator,
                        nb_val_samples=len(validation_log_lines))

    model.save('model.h5')
