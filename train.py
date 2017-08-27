import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


BATCH_SIZE = 64
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


def data_generator(log_lines, batch_size, read_from_log_line):
    while True:  # fit_generator requires this generator to have infinite loop
        shuffled_log_lines = shuffle(log_lines)  # Shuffle data before each epoch
        for line_offset in range(0, len(shuffled_log_lines), batch_size):
            batch_log_lines = shuffled_log_lines[line_offset: line_offset + batch_size]
            images = []
            steerings = []
            for line in batch_log_lines:
                read_images, read_steerings = read_from_log_line(line)
                images.extend(read_images)
                steerings.extend(read_steerings)

            X_train = np.array(images)
            y_train = np.array(steerings)
            yield shuffle(X_train, y_train)  # Shuffle each batch's internal data


def read_training_data_from_log_line(line):
    images = []
    steerings = []

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

    return images, steerings


def read_validation_data_from_log_line(line):
    images = []
    steerings = []

    # only use the center image because we will only use ground truth in validation data
    center_image_path = line[0]
    image = cv2.imread(center_image_path)
    image = process_image(image)
    steering = float(line[3])
    images.append(image)
    steerings.append(steering)

    return images, steerings


if __name__ == "__main__":
    log_lines = read_driving_log('data_1/driving_log.csv')
    training_log_lines, validation_log_lines = train_test_split(log_lines, test_size=0.2)

    training_data_generator = data_generator(training_log_lines, BATCH_SIZE,
                                             read_training_data_from_log_line)
    validation_data_generator = data_generator(validation_log_lines, BATCH_SIZE,
                                               read_validation_data_from_log_line)

    # for batch in training_data_generator:
    #     print(batch[0].shape)
    #     print(batch[1].shape)
    #     print("")
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
    model.fit_generator(training_data_generator, len(training_log_lines) * 3, nb_epoch=EPOCH_COUNT,
                        validation_data=validation_data_generator,
                        nb_val_samples=len(validation_log_lines))

    model.save('model.h5')
