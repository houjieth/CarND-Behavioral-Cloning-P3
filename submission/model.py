import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import click
import sys

from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Activation
from keras.layers.convolutional import Convolution2D

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


BATCH_SIZE = 64
EPOCH_COUNT = 5
CAMERA_STEERING_CORRECTION = 0.5


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

    # print(line)
    center_image_path, left_image_path, right_image_path = line[0], line[1], line[2]

    # center image
    image = cv2.imread(center_image_path)
    image = process_image(image)
    steering = float(line[3])
    images.append(image)
    steerings.append(steering)

    # center image flipped
    image = cv2.imread(center_image_path)
    image = np.fliplr(image)
    image = process_image(image)
    steering = -float(line[3])
    images.append(image)
    steerings.append(steering)

    # left image
    image = cv2.imread(left_image_path)
    image = process_image(image)
    steering = float(line[3]) + CAMERA_STEERING_CORRECTION
    images.append(image)
    steerings.append(steering)

    # right image
    image = cv2.imread(right_image_path)
    image = process_image(image)
    steering = float(line[3]) - CAMERA_STEERING_CORRECTION
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


def create_model():
    model = Sequential()

    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    print(model.output_shape)

    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    print(model.output_shape)

    model.add(Convolution2D(24, 5, 5, subsample=(2, 2)))
    print(model.output_shape)

    model.add(Convolution2D(36, 5, 5, subsample=(2, 2)))
    print(model.output_shape)

    model.add(Convolution2D(48, 5, 5, subsample=(2, 2)))
    print(model.output_shape)

    model.add(Convolution2D(64, 3, 3))
    print(model.output_shape)

    model.add(Convolution2D(64, 3, 3))
    print(model.output_shape)

    model.add(Flatten())
    print(model.output_shape)

    model.add(Dense(500))
    print(model.output_shape)

    model.add(Dense(100))
    print(model.output_shape)

    model.add(Dense(50))
    print(model.output_shape)

    model.add(Dense(10))
    print(model.output_shape)

    model.add(Dense(1))
    print(model.output_shape)

    model.compile(loss='mse', optimizer='adam')

    return model


@click.command()
@click.option('--model_file_input_path', help='File path of the Model which this training will be based on')
@click.option('--model_file_output_path', help='File path of the output model')
@click.option('--driving_log_csv_file_path', help='File path to driving_log.csv')
def train(model_file_input_path, model_file_output_path, driving_log_csv_file_path):
    if not model_file_output_path:
        sys.exit('Missing model_file_output_path. Use --help')
    if not driving_log_csv_file_path:
        sys.exit('Missing driver_log_csv_file_path. Use --help')

    if model_file_input_path:
        model = load_model(model_file_input_path)
    else:
        model = create_model()

    log_lines = read_driving_log(driving_log_csv_file_path)
    training_log_lines, validation_log_lines = train_test_split(log_lines, test_size=0.2)

    training_data_generator = data_generator(training_log_lines, BATCH_SIZE,
                                             read_training_data_from_log_line)
    validation_data_generator = data_generator(validation_log_lines, BATCH_SIZE,
                                               read_validation_data_from_log_line)

    history_object = model.fit_generator(training_data_generator, len(training_log_lines) * 4, nb_epoch=EPOCH_COUNT,
                        validation_data=validation_data_generator,
                        nb_val_samples=len(validation_log_lines))

    print(history_object.history.keys())

    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

    model.save(model_file_output_path)


if __name__ == "__main__":
    train()


