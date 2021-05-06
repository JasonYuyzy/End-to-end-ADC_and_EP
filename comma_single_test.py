from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable
import torchvision.transforms as transforms

import h5py
import pylab as plt
import cv2
import numpy as np
import pandas as pd
import argparse

from model_three import *

import time

"""dataset reader"""

def read_from_h5(camera_names):
    logs_names = [x.replace('camera', 'log') for x in camera_names]

    angle = []
    c5x = []
    camera_file = []
    filters = []
    lastidx = 0
    time_len = 1

    """read the data file"""

    for came_word, log_word in zip(camera_names, logs_names):
        try:
            with h5py.File(log_word, "r") as t5:
                camera_5 = h5py.File(came_word, "r")
                camera_file.append(camera_5)
                image_set = camera_5["X"]
                c5x.append((lastidx, lastidx + image_set.shape[0], image_set))

                steering_angle_set = t5["steering_angle"][:]

                idxs = np.linspace(0, steering_angle_set.shape[0] - 1, image_set.shape[0]).astype("int")  # approximate alignment
                angle.append(steering_angle_set[idxs])

                goods = np.abs(angle[-1]) <= 200

                filters.append(np.argwhere(goods)[time_len - 1:] + (lastidx + time_len - 1))
                lastidx += goods.shape[0]
                # check for mismatched length bug
                print("image_set {} | t {} | f {}".format(image_set.shape[0], steering_angle_set.shape[0],goods.shape[0]))
                if image_set.shape[0] != angle[-1].shape[0] or image_set.shape[0] != goods.shape[0]:
                    raise Exception("bad shape")

        except IOError:
            import traceback
            traceback.print_exc()
            print("failed to open", log_word)

    angle = np.concatenate(angle, axis=0)
    filters = np.concatenate(filters, axis=0).ravel()

    return c5x, angle, filters, camera_file

"""speed difference"""

def speed_dis(speeds):
    pre_speed = speeds[0]
    max_dif = 0
    dif_list = list()
    print(len(speeds))
    for i in range(1, len(speeds)-1):
        n_speed = speeds[i]
        dif = n_speed - pre_speed
        pre_speed = n_speed
        if abs(speeds[i]) > 0.7:
            dif_list.append(dif)
        if abs(dif) > abs(max_dif):
            max_dif = dif

    print(max_dif)
    print(len(dif_list))
    exit()


"""load image data"""


def augment(img):
    current_image = img.swapaxes(0,2).swapaxes(0,1)

    '''if np.random.rand() < 0.5:
        # left-right transfer
        current_image = cv2.flip(current_image, 1)
        angle = angle * -1.0'''

    # random brightness
    '''current_image = cv2.cvtColor(current_image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    current_image[:, :, 2] = current_image[:, :, 2] * ratio
    current_image = cv2.cvtColor(current_image, cv2.COLOR_HSV2RGB)'''

    # [height, width, deep] crop
    current_image = current_image[20:-10, :, :]  # remove the sky and the car front

    current_image = cv2.resize(current_image, (227, 227), cv2.INTER_AREA)
    # rgb2yuv
    #current_image = cv2.cvtColor(current_image, cv2.COLOR_RGB2YUV)

    return current_image


def image_loader(image_set, angle_list):
    img_list = list()
    steer_list = list()

    for img, steering in zip(image_set, angle_list):
        img = augment(img)
        img_list.append(img)
        steer_list.append(steering)

    return img_list, steer_list


"""Sequence generator"""
'''def speed_class(speeds):
    speed_classes = []
    for i in range(len(speeds)):
        # speed classes, speed states
        if speeds[i] > 0.7:
            speed_classes.append([0, 0, 1])  # speed-up
        elif speeds[i] < -0.7:
            speed_classes.append([1, 0, 0])  # speed-down
        else:
            speed_classes.append([0, 1, 0])  # keeping

    a = {'speed_classes': speed_classes}
    speed_generate = pd.DataFrame(a)
    print("Speed classes generated")
    return speed_generate'''

def speed_sequence(speeds):
    speed_sequence = []
    i_sequence = [speeds[0] for i in range(args.sequence_length)]
    for i in range(len(speeds)):
        # speed sequence 10 speeds
        if i == len(speeds) - 1:
            i_sequence.pop(args.sequence_length-1)
            i_sequence.insert(0, speeds[i])
        else:
            i_sequence.pop(args.sequence_length-1)
            i_sequence.insert(0, speeds[i + 1])
        sq = i_sequence.copy()
        speed_sequence.append(sq)

    a = {'speed_sequence': speed_sequence}
    speed_generate = pd.DataFrame(a)
    print("Speed sequence generated")
    return speed_generate

def steering_generator(steerings):
    steer_sequence = []
    s_sequence = [steerings[0] for i in range(args.sequence_length)]
    for i in range(len(steerings)):
        # steer sequence batch speeds
        if i == len(steerings) - 1:
            s_sequence.pop(args.sequence_length - 1)
            s_sequence.insert(0, steerings[i])
        else:
            s_sequence.pop(args.sequence_length - 1)
            s_sequence.insert(0, steerings[i + 1])
        sq = s_sequence.copy()
        steer_sequence.append(sq)

    a = {'steer_sequence': steer_sequence}
    steer_generate = pd.DataFrame(a)

    return steer_generate

"""Raw data process"""

def data_loader(images, steers):
    print("Processing the raw data...")
    # generate speed sequence and classes
    processed_images, precessed_steers = image_loader(images, steers)
    print("Image processed")

    data_df = {'img': processed_images, 'steering': precessed_steers}
    dataframe = pd.DataFrame(data_df)

    dataframe = dataframe.drop(dataframe[(dataframe['steering'] < -720) | (dataframe['steering'] > 720)].index)

    processed_images = dataframe['img'].values
    steers_new = dataframe['steering'].values

    precessed_steers = np.append(steers_new, [-720, 720])
    precessed_steers = steer_scaler.fit_transform(precessed_steers.reshape(-1, 1))
    precessed_steers = np.delete(precessed_steers, [len(precessed_steers) - 2, len(precessed_steers) - 1])
    print("Normalized steers")


    steer_generate = steering_generator(precessed_steers)
    steering_sequence = steer_generate['steer_sequence'].values
    print("Steer sequence generated")


    return processed_images, steering_sequence, steers_new



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Self_driving car model parameters with (MESLoss) train with comma ai dataset in batch 256')
    parser.add_argument('--image_size', type=set, default=(227, 227), help="image size for training")
    parser.add_argument('--validation_data_size', type=float, default=0.2, help="image size for training")
    parser.add_argument('--model_predict', type=str, default='./models/model_comma_single.pth', help="trained model for steering and speed prediction")

    parser.add_argument('--lstm_layers_num', type=int, default=2, help="num of lstm")
    parser.add_argument('--lstm_hidden_layers', type=int, default=32, help="num of hidden of lstm")

    parser.add_argument('--batch_size', type=int, default=32, help="batch size for training")
    parser.add_argument('--sequence_length', type=int, default=32, help="batch size for training")
    parser.add_argument('--training_range', type=int, default=None, help="range of data cycle for training")
    parser.add_argument('--validation_range', type=int, default=None, help="range of data cycle for validation")

    parser.add_argument('--lr_rate', type=float, default=1e-5, help="learning rate")
    parser.add_argument('--train_epoch', type=int, default=10, help="training epochs")
    parser.add_argument('--weighted_loss', type=bool, default=True, help="whether weight the loss")
    parser.add_argument('--lambda_value_speed', type=float, default=0.7, help="loss weight for speed loss")
    parser.add_argument('--lambda_value_failure', type=float, default=0.0, help="loss weight for failure loss")
    parser.add_argument('--trained', type=bool, default=False, help="continue train or not")
    parser.add_argument('--loss_function', type=str, default="Mean", help="chose loss function")
    parser.add_argument('--loss_display', type=int, default=700, help="display the loss for every n steps")

    parser.add_argument('--loss_image_path', type=str, default='model_comma_single.png', help="loss image saver")

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device running:", device)

    # standardize speed sequence
    steer_scaler = MinMaxScaler(feature_range=(-1, 1))
    speed_scaler = MinMaxScaler(feature_range=(-1, 1))

    camera_names = [
        './camera/2016-02-08--14-56-28.h5'
    ]

    """**loss-function**"""

    criterion = nn.MSELoss()

    checkpoint = torch.load(args.model_predict, map_location=device)
    model_predict = checkpoint['model']
    model_predict.eval()
    model_predict.hidden_cell_sp = (torch.zeros(args.lstm_layers_num, 1, args.lstm_hidden_layers).to(device), torch.zeros(args.lstm_layers_num, 1, args.lstm_hidden_layers).to(device))
    model_predict.hidden_cell_st = (torch.zeros(args.lstm_layers_num, 1, args.lstm_hidden_layers).to(device), torch.zeros(args.lstm_layers_num, 1, args.lstm_hidden_layers).to(device))
    print(model_predict)

    transformations = transforms.Compose([transforms.ToTensor()])

    number = 0

    for name in camera_names:
        steering_prediction_list = list()
        steer_ground_list = list()

        c5x, angle, filters, camera_file = read_from_h5([name])

        model_name = name

        # speed_dis(accel)
        processed_images, steering_sequence, steers_new = data_loader(c5x[0][2], angle)

        steer_scaler.fit_transform(np.array([-720, 720]).reshape(-1, 1))

        """training data generation"""
        print("Starting testing...")
        for i in range(len(processed_images)):

            image = transformations(processed_images[i])
            image = image.view(1, 3, 227, 227)
            image = Variable(image)

            image = image.float().to(device)

            output_steering = model_predict(image)

            if device =='cuda':
                p_steer = output_steering.view(-1).cpu().data.numpy()
                p_steer = steer_scaler.inverse_transform(p_steer.reshape(-1, 1))[0]
            else:
                p_steer = output_steering.view(-1).data.numpy()
                p_steer = steer_scaler.inverse_transform(p_steer.reshape(-1, 1))[0]


            steering_prediction_list.append(p_steer)
            steer_ground_list.append(steers_new[1])

        x = range(len(steering_prediction_list))

        fig = plt.figure()
        ax = fig.add_subplot(111)
        lns1 = ax.plot(x, steering_prediction_list, '-', label='prediction steer')
        lns2 = ax.plot(x, steer_ground_list, '-', label='ground truth steer')
        # added these two lines
        lns = lns1 + lns2
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc=0)
        ax.grid()
        ax.set_xlabel("numbers")
        ax.set_ylabel("degree")

        plt.savefig("./image/steering_compare{}".format(number) + args.loss_image_path)
        plt.close()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        lns1 = ax.plot(x[2000:3000], steering_prediction_list[2000:3000], '-', label='prediction steer')
        lns2 = ax.plot(x[2000:3000], steers_new[2000:3000], '-', label='ground truth steer')
        # added these two lines
        lns = lns1 + lns2
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc=0)
        ax.grid()
        ax.set_xlabel("numbers")
        ax.set_ylabel("degree")

        plt.savefig("./image/steering_2-3compare{}".format(number) + args.loss_image_path)
        plt.close()

        print(f"Steer_MSE：{mean_squared_error(steering_prediction_list, steers_new)}")
        print(f"Steer_MAE：{mean_absolute_error(steering_prediction_list, steers_new)}")
        print(f"Steer_RMSE：{np.sqrt(mean_squared_error(steering_prediction_list, steers_new))}")
        print(f"Steer_R^2：{r2_score(steering_prediction_list, steers_new)}")

        print(f"Steer_MAE(range2000-3000)：{mean_absolute_error(steering_prediction_list[2000:3000], steers_new[2000:3000])}")


        number += 1