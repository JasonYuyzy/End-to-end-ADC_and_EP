import torch.optim as optim
from torch.utils import data
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

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
    speed = []
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

                speed_value = t5["speed"][:]
                steering_angle_set = t5["steering_angle"][:]

                idxs = np.linspace(0, steering_angle_set.shape[0] - 1, image_set.shape[0]).astype("int")  # approximate alignment
                angle.append(steering_angle_set[idxs])
                speed.append(speed_value[idxs])

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
    speed = np.concatenate(speed, axis=0)
    filters = np.concatenate(filters, axis=0).ravel()

    return c5x, angle, speed, filters, camera_file

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


def augment(img, angle):
    current_image = img.swapaxes(0,2).swapaxes(0,1)

    if np.random.rand() < 0.5:
        # left-right transfer
        current_image = cv2.flip(current_image, 1)
        angle = angle * -1.0

    # random brightness
    current_image = cv2.cvtColor(current_image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    current_image[:, :, 2] = current_image[:, :, 2] * ratio
    current_image = cv2.cvtColor(current_image, cv2.COLOR_HSV2RGB)

    # [height, width, deep] crop
    current_image = current_image[20:-10, :, :]  # remove the sky and the car front

    current_image = cv2.resize(current_image, (227, 227), cv2.INTER_AREA)
    # rgb2yuv
    current_image = cv2.cvtColor(current_image, cv2.COLOR_RGB2YUV)

    return current_image, angle


def image_loader(image_set, angle_list):
    img_list = list()
    steer_list = list()

    for img, steering in zip(image_set, angle_list):

        img, steering_angle = augment(img, steering)
        img_list.append(img)
        steer_list.append(steering_angle)

    return img_list, steer_list


"""Sequence generator"""
def speed_class(speeds):
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
    return speed_generate

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

def data_loader(images, steers, speeds):
    print("Processing the raw data...")
    # generate speed sequence and classes
    processed_images, precessed_steers = image_loader(images, steers)
    print("Image processed")

    data_df = {'img': processed_images, 'steering': precessed_steers, 'speed': speeds}
    dataframe = pd.DataFrame(data_df)

    dataframe = dataframe.drop(dataframe[(dataframe['steering'] < -720) | (dataframe['steering'] > 720)].index)

    processed_images = dataframe['img'].values
    steers_new = dataframe['steering'].values
    speeds_new = dataframe['speed'].values


    precessed_steers = np.append(steers_new, [-720, 720])
    precessed_steers = steer_scaler.fit_transform(precessed_steers.reshape(-1, 1))
    precessed_steers = np.delete(precessed_steers, [len(precessed_steers) - 2, len(precessed_steers) - 1])
    print("Normalized steers")


    precessed_speeds = np.append(speeds_new, [0, 35])
    precessed_speeds = speed_scaler.fit_transform(precessed_speeds.reshape(-1, 1))
    precessed_speeds = np.delete(precessed_speeds, [len(precessed_speeds) - 2, len(precessed_speeds) - 1])
    print("normalized speeds")

    '''precessed_steers = np.append(steers, [-5300, 5300])
    precessed_steers = steer_scaler.fit_transform(precessed_steers.reshape(-1, 1))
    precessed_steers = np.delete(precessed_steers, [len(precessed_steers) - 2, len(precessed_steers) - 1])
    print("Normalized steers")

    precessed_speeds = np.append(speeds, [0, 35])
    precessed_speeds = speed_scaler.fit_transform(precessed_speeds.reshape(-1, 1))
    precessed_speeds = np.delete(precessed_speeds, [len(precessed_speeds) - 2, len(precessed_speeds) - 1])
    print("normalized speeds")'''


    steer_generate = steering_generator(precessed_steers)
    steering_sequence = steer_generate['steer_sequence'].values
    print("Steer sequence generated")

    speed_sequence_generate = speed_sequence(precessed_speeds)
    speed_sequences = speed_sequence_generate['speed_sequence'].values
    print("Speed sequence generated")


    train_images, valid_images, train_steer_sequence, valid_steer_sequence = train_test_split(processed_images[:], steering_sequence, test_size=args.validation_data_size, random_state=42, shuffle=False)
    train_sequence, valid_sequence = train_test_split(speed_sequences, test_size=args.validation_data_size, random_state=42, shuffle=False)


    return train_images, valid_images, train_sequence, valid_sequence, train_steer_sequence, valid_steer_sequence


"""training dataset generator"""
class Dataset(data.Dataset):
    def __init__(self, images, speed_sequences, steer_sequence, transforms=None):
        self.images = images
        self.speed_sequences = speed_sequences
        self.steer_sequences = steer_sequence
        self.transform = transforms

    def __getitem__(self, index):
        img = self.images[index]
        batch_speed_sequence = self.speed_sequences[index]
        batch_steer_sequence = self.steer_sequences[index]

        # steering_angle = float(steer)

        speed_sq = np.array(batch_speed_sequence)
        speed_sq = speed_sq.reshape((args.sequence_length, 1))

        steer_sq = np.array(batch_steer_sequence)
        steer_sq = steer_sq.reshape((args.sequence_length, 1))


        img = self.transform(img)

        return (img, steer_sq[1], speed_sq, speed_sq[1], steer_sq)

    # function return the len of the training sets
    def __len__(self):
        return len(self.images)

"""helper function to make getting another batch of data easier"""

def cycle(iterable):
    while True:
        for x in iterable:
            yield x


"""train function"""

def train(train_iterator, valid_iterator, model_name):
    epoch_trained = 0

    if args.trained:
        checkpoint = torch.load(args.model_step_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        last_dataset = checkpoint['name']
        epoch_trained = checkpoint['epoch']
        print("Last dataset:", last_dataset)
        print("Continue train from epoch:", epoch_trained)

    for epoch in range(1, args.train_epoch):
        model.to(device)
        print("Epoch:", epoch + epoch_trained)

        train_loss = 0.0
        train_steer_loss = 0.0
        train_speed_loss = 0.0

        model.train()
        for local_batch_t in range(args.training_range):
            imgs, steering_angles, speed_sqs, speed_cls, steer_sqs = next(train_iterator)

            img_inputs, steer_labels, speed_inputs, speed_labels, steer_inputs = imgs.float().to(device), steering_angles.float().to(device), speed_sqs.float().to(device), speed_cls.float().to(device), steer_sqs.float().to(device)

            # optimizer zero gradient
            optimizer.zero_grad()

            model.hidden_cell_sp = (torch.zeros(args.lstm_layers_num, args.batch_size, args.lstm_hidden_layers).to(device), torch.zeros(args.lstm_layers_num, args.batch_size, args.lstm_hidden_layers).to(device))
            model.hidden_cell_st = (torch.zeros(args.lstm_layers_num, args.batch_size, args.lstm_hidden_layers).to(device), torch.zeros(args.lstm_layers_num, args.batch_size, args.lstm_hidden_layers).to(device))

            # training
            output_steering, output_speeds = model(img_inputs, speed_inputs, steer_inputs)


            steer_loss = criterion_steer(output_steering, steer_labels)
            speed_loss = criterion_speed(output_speeds, speed_labels)

            if args.weighted_loss:
                loss = (1-args.lambda_value_speed) * steer_loss + args.lambda_value_speed * speed_loss
            else:
                loss = steer_loss + speed_loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_steer_loss += steer_loss.item()
            train_speed_loss += speed_loss.item()

            if local_batch_t % args.loss_display == 0:
                print('Speed Loss: %.3f ' % (speed_loss))
                print('Steer Loss: %.3f ' % (steer_loss))
                print('Total Loss: %.3f ' % (train_loss / (local_batch_t + 1)))
                print('\n')

        LSTM_Loss_List.append(train_loss / (local_batch_t + 1))
        steer_loss_list.append(train_steer_loss / (local_batch_t + 1))
        speed_loss_list.append(train_speed_loss / (local_batch_t + 1))

        # validation
        valid_loss = 0.0
        valid_steer_loss = 0.0
        valid_speed_loss = 0.0

        model.eval()
        with torch.set_grad_enabled(False):
            for local_batch_v in range(args.validation_range):
                v_imgs, v_steering_angles, v_speed_sqs, v_speed_cls, v_steer_sqs = next(valid_iterator)

                v_img_inputs, v_steer_labels, v_speed_inputs, v_speed_labels, v_steer_inputs = v_imgs.float().to(device), v_steering_angles.float().to(device), v_speed_sqs.float().to(device), v_speed_cls.float().to(device), v_steer_sqs.float().to(device)

                # optimizer zero gradient
                optimizer.zero_grad()

                v_output_steering, v_output_speeds = model(v_img_inputs, v_speed_inputs, v_steer_inputs)

                va_steer_loss = criterion_steer(v_output_steering, v_steer_labels)
                va_speed_loss = criterion_speed(v_output_speeds, v_speed_labels)
                if args.weighted_loss:
                    va_loss = (1-args.lambda_value_speed) * va_steer_loss + args.lambda_value_speed * va_speed_loss
                else:
                    va_loss = va_steer_loss + va_speed_loss

                valid_loss += va_loss.item()
                valid_steer_loss += va_steer_loss.item()
                valid_speed_loss += va_speed_loss.item()

            Validation_Loss_List.append(valid_loss / (local_batch_v + 1))
            Vsteer_Loss_List.append(valid_loss / (local_batch_v + 1))
            Vspeed_Loss_List.append(valid_loss / (local_batch_v + 1))

            print('Speed Valid Loss: %.3f ' % (va_speed_loss))
            print('Steer Valid Loss: %.3f ' % (va_steer_loss))
            print('Valid Loss: %.3f ' % (valid_loss / (local_batch_v + 1)))
            print('\n')

        if epoch % 100 == 0:
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch + epoch_trained, 'name': model_name}
            torch.save(state, args.model_step_path)
            print("Model saved in epoch:", epoch)

    """train loss in epoches"""
    """sum loss (weighted)"""
    x1 = range(len(LSTM_Loss_List))
    y1 = LSTM_Loss_List
    x2 = range(len(Validation_Loss_List))
    y2 = Validation_Loss_List

    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.plot(x1, y1, '.-')
    plt.xlabel('epoches')
    plt.ylabel('Train loss')

    plt.subplot(122)
    plt.plot(x2, y2, '.-')
    plt.xlabel('epoches')
    plt.ylabel('Validation loss')

    plt.savefig("./image/total_loss_" + args.loss_image_path)

    """steer loss"""

    x1 = range(len(steer_loss_list))
    y1 = steer_loss_list
    x2 = range(len(Vsteer_Loss_List))
    y2 = Vsteer_Loss_List

    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.plot(x1, y1, '.-')
    plt.xlabel('epoches')
    plt.ylabel('Train loss')

    plt.subplot(122)
    plt.plot(x2, y2, '.-')
    plt.xlabel('epoches')
    plt.ylabel('Validation loss')

    plt.savefig("./image/steer_loss_" + args.loss_image_path)

    """speed loss"""

    x1 = range(len(speed_loss_list))
    y1 = speed_loss_list
    x2 = range(len(Vspeed_Loss_List))
    y2 = Vspeed_Loss_List

    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.plot(x1, y1, '.-')
    plt.xlabel('epoches')
    plt.ylabel('Train loss')

    plt.subplot(122)
    plt.plot(x2, y2, '.-')
    plt.xlabel('epoches')
    plt.ylabel('Validation loss')

    plt.savefig("./image/speed_loss_" + args.loss_image_path)

    return model



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Self_driving car model parameters with (MESLoss) train with comma ai dataset in batch 256')
    parser.add_argument('--image_size', type=set, default=(227, 227), help="image size for training")
    parser.add_argument('--validation_data_size', type=float, default=0.2, help="image size for training")
    parser.add_argument('--GPU_device', type=bool, default=None, help="Use GPU or not")

    parser.add_argument('--lstm_layers_num', type=int, default=2, help="num of lstm")
    parser.add_argument('--lstm_hidden_layers', type=int, default=32, help="num of hidden of lstm")

    parser.add_argument('--batch_size', type=int, default=32, help="batch size for training")
    parser.add_argument('--sequence_length', type=int, default=32, help="batch size for training")
    parser.add_argument('--training_range', type=int, default=None, help="range of data cycle for training")
    parser.add_argument('--validation_range', type=int, default=None, help="range of data cycle for validation")

    parser.add_argument('--lr_rate', type=float, default=1e-5, help="learning rate")
    parser.add_argument('--train_epoch', type=int, default=61, help="training epochs")
    parser.add_argument('--weighted_loss', type=bool, default=True, help="whether weight the loss")
    parser.add_argument('--lambda_value_speed', type=float, default=0.7, help="loss weight for speed loss")
    parser.add_argument('--lambda_value_failure', type=float, default=0.0, help="loss weight for failure loss")
    parser.add_argument('--trained', type=bool, default=False, help="continue train or not")
    parser.add_argument('--loss_function', type=str, default="Mean", help="chose loss function")
    parser.add_argument('--loss_display', type=int, default=700, help="display the loss for every n steps")

    parser.add_argument('--loss_image_path', type=str, default='comma_adc_train.png', help="loss image saver")
    parser.add_argument('--model_step_path', type=str, default='./models/model_comma_adc_train.h5', help="step saver model path")
    parser.add_argument('--model_final_path', type=str, default='./models/model_comma_adc_train.pth', help="final model path")

    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
        args.GPU_device = True
    else:
        device = torch.device('cpu')
        args.GPU_device = False
    print("Device running:", device)

    # standardize speed sequence
    steer_scaler = MinMaxScaler(feature_range=(-1, 1))
    speed_scaler = MinMaxScaler(feature_range=(-1, 1))

    # datasets name
    camera_names = [
        './camera/2016-06-08--11-46-01.h5'
        './camera/2016-01-31--19-19-25.h5',
        #'./camera/2016-02-08--14-56-28.h5'
    ]

    """**loss-function**"""

    model = CNNLSTMpredict_straight()
    criterion_speed = nn.MSELoss()
    criterion_steer = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr_rate)

    LSTM_Loss_List = list()
    steer_loss_list = list()
    speed_loss_list = list()

    Validation_Loss_List = list()
    Vsteer_Loss_List = list()
    Vspeed_Loss_List = list()

    count = 0

    for name in camera_names:
        c5x, angle, speed, filters, camera_file = read_from_h5([name])

        # speed_dis(accel)
        train_images, valid_images, train_sequence, valid_sequence, train_steer_sequence, valid_steer_sequence = data_loader(c5x[0][2], angle, speed)

        """training data generation"""

        # transforms
        transformations = transforms.Compose([transforms.ToTensor()])
        # separate and load training data
        training_set = Dataset(train_images, train_sequence, train_steer_sequence, transformations)
        training_generator = data.DataLoader(training_set, shuffle=True, batch_size=args.batch_size, drop_last=True)
        # separate and load validation data
        validation_set = Dataset(valid_images, valid_sequence, valid_steer_sequence, transformations)
        validation_generator = data.DataLoader(validation_set, shuffle=True, batch_size=args.batch_size, drop_last=True)

        args.training_range = len(training_generator)
        args.validation_range = len(validation_generator)

        t_iterator = iter(cycle(training_generator))
        v_iterator = iter(cycle(validation_generator))

        # start training
        model = train(train_iterator=t_iterator, valid_iterator=v_iterator, model_name=name.split('.')[0])

        args.trained = True

        ###model saver
        state = {
            'model': model.module if device == 'cuda' else model,
        }
        print(state)
        torch.save(state, args.model_final_path, _use_new_zipfile_serialization=False)