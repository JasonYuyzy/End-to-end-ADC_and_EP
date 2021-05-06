import torch.optim as optim
from torch.utils import data
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time

from model_three import *
import argparse
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1, 1))


"""**load image data**"""


def augment(imgName, angle):
    name = args.image_path + imgName.split('/')[-1]
    current_image = cv2.imread(name)

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
    current_image = current_image[50:-23, :, :]  # remove the sky and the car front
    # resize
    current_image = cv2.resize(current_image, args.image_size, cv2.INTER_AREA)
    # rgb2yuv
    current_image = cv2.cvtColor(current_image, cv2.COLOR_RGB2YUV)

    return current_image, angle

def image_loader(image_pth, angle_list):
    img_list = list()
    steer_list = list()
    """testing"""
    for im_path, steering in zip(image_pth, angle_list):
        # three images center left and right
        choice = np.random.choice(3)
        if choice == 0:  # left image
            if steering > 0.8:
                img, steering = augment(im_path[1], steering)
            else:
                img, steering = augment(im_path[1], steering + 0.2)
        elif choice == 1:  # right image
            if steering < -0.8:
                img, steering = augment(im_path[2], steering)
            else:
                img, steering = augment(im_path[2], steering - 0.2)
        else:  # center image
            img, steering = augment(im_path[0], steering)

        # center image
        #img, steering = augment(im_path[0], steering)

        img_list.append(img)
        steer_list.append(steering)

    return img_list, steer_list


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


def speed_dis(speeds):
    pre_speed = speeds[0]
    dif_list = []
    for i in range(1, len(speeds)-1):
        n_speed = speeds[i]
        dif_list.append(pre_speed - n_speed)
        pre_speed = n_speed

    print(np.mean(dif_list))
    exit()


"""**data-set collection**"""

def data_loader():
    data_df = pd.read_csv(args.log_path, header=None)
    images = data_df[[0, 1, 2]].values
    speeds = data_df[6].values
    steers = data_df[3].values

    #speed_dis(speeds
    # load image data
    images, steers = image_loader(images, steers)
    print("Image processed")

    if not args.speed_data_original:
    # normalize the speed to range -1 ~ 1 from range 0 ~ 32
        speeds = np.append(speeds, [0, 32])
        speeds = scaler.fit_transform(speeds.reshape(-1, 1))
        speeds = np.delete(speeds, [len(speeds) - 2, len(speeds) - 1])
        print("normalized speeds")

    steer_generate = steering_generator(steers)
    steering_sequence = steer_generate['steer_sequence'].values
    print("Steer sequence generated")

    speed_sequence_generate = speed_sequence(speeds)
    speed_sequences = speed_sequence_generate['speed_sequence'].values
    print("Speed sequence generated")


    train_images, valid_images, train_steer_sequence, valid_steer_sequence = train_test_split(images, steering_sequence, test_size=args.validation_data_size, random_state=42, shuffle=False)
    train_speed_sequence, valid_speed_sequence = train_test_split(speed_sequences, test_size=args.validation_data_size, random_state=42, shuffle=False)


    return train_images, valid_images, train_speed_sequence, valid_speed_sequence, train_steer_sequence, valid_steer_sequence



"""data preparetion"""

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

        speed_sq = np.array(batch_speed_sequence)
        speed_sq = speed_sq.reshape((args.sequence_length, 1))

        steer_sq = np.array(batch_steer_sequence)
        steer_sq = steer_sq.reshape((args.sequence_length, 1))

        if args.speed_data_original:
            speed_label = np.append(speed_sq[1], [0, 32])
            speed_label = scaler.fit_transform(speed_label.reshape(-1, 1))
            speed_label = np.delete(speed_label, [len(speed_label) - 2, len(speed_label) - 1])

        img = self.transform(img)

        if args.speed_data_original:
            return (img, steer_sq[1], speed_sq, speed_label, steer_sq)
        else:
            return (img, steer_sq[1], speed_sq, speed_sq[1], steer_sq)

    # function return the len of the training sets
    def __len__(self):
        return len(self.images)


"""helper function to make getting another batch of data easier"""

def cycle(iterable):
    while True:
        for x in iterable:
            yield x


"""**train function**"""
def train(train_iterator, valid_iterator, model):
    LSTM_Loss_List = list()
    steer_loss_list = list()
    speed_loss_list = list()

    Validation_Loss_List = list()
    Vsteer_Loss_List = list()
    Vspeed_Loss_List = list()

    epoch_trained = 0

    if args.trained:
        checkpoint = torch.load(args.model_step_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch_trained = checkpoint['epoch']

    for epoch in range(args.train_epoch):
        model.to(device)
        print("Epoch:", epoch + epoch_trained)

        train_loss = 0.0
        train_steer_loss = 0.0
        train_speed_loss = 0.0

        model.train()
        for local_batch_t in range(args.training_range):
            imgs, steering_angles, speed_sqs, speed_cls, steer_sqs = next(train_iterator)

            # To torch
            img_inputs, steer_labels, speed_inputs, speed_labels, steer_inputs = imgs.float().to(device), steering_angles.float().to(device), speed_sqs.float().to(device), speed_cls.float().to(device), steer_sqs.float().to(device)

            #optimizer zero gradient
            optimizer.zero_grad()

            model.hidden_cell_sp = (torch.zeros(args.lstm_layers_num, args.batch_size, args.lstm_hidden_layers).to(device), torch.zeros(args.lstm_layers_num, args.batch_size, args.lstm_hidden_layers).to(device))
            model.hidden_cell_st = (torch.zeros(args.lstm_layers_num, args.batch_size, args.lstm_hidden_layers).to(device), torch.zeros(args.lstm_layers_num, args.batch_size, args.lstm_hidden_layers).to(device))

            #training
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
                print('Speed Loss: %.3f '% (speed_loss))
                print('Steer Loss: %.3f '% (steer_loss))
                print('Total Loss: %.3f '% (train_loss/(local_batch_t+1)))
                print('\n')

        LSTM_Loss_List.append(train_loss/(local_batch_t+1))
        steer_loss_list.append(train_steer_loss/(local_batch_t+1))
        speed_loss_list.append(train_speed_loss/(local_batch_t+1))

        #validation
        valid_loss = 0.0
        valid_steer_loss = 0.0
        valid_speed_loss = 0.0

        model.eval()
        with torch.set_grad_enabled(False):
            for local_batch_v in range(args.validation_range):
                v_imgs, v_steering_angles, v_speed_sqs, v_speed_cls, v_steer_sqs = next(valid_iterator)

                v_img_inputs, v_steer_labels, v_speed_inputs, v_speed_labels, v_steer_inputs = v_imgs.float().to(device), v_steering_angles.float().to(device), v_speed_sqs.float().to(device), v_speed_cls.float().to(device), v_steer_sqs.float().to(device)

                #optimizer zero gradient
                optimizer.zero_grad()

                v_output_steering, v_output_speeds = model(v_img_inputs, v_speed_inputs, v_steer_inputs)

                #output_steering = model(imgs)
                va_steer_loss = criterion_steer(v_output_steering, v_steer_labels)
                va_speed_loss = criterion_speed(v_output_speeds, v_speed_labels)

                if args.weighted_loss:
                    va_loss = (1-args.lambda_value_speed) * va_steer_loss + args.lambda_value_speed * va_speed_loss
                else:
                    va_loss = va_steer_loss + va_speed_loss

                valid_loss += va_loss.item()
                valid_steer_loss += va_steer_loss.item()
                valid_speed_loss += va_speed_loss.item()

            Validation_Loss_List.append(valid_loss/(local_batch_v+1))
            Vsteer_Loss_List.append(valid_steer_loss/(local_batch_v+1))
            Vspeed_Loss_List.append(valid_speed_loss/(local_batch_v+1))


            print('Speed Valid Loss: %.3f '% (va_speed_loss))
            print('Steer Valid Loss: %.3f '% (va_steer_loss))
            print('Valid Loss: %.3f '% (valid_loss/(local_batch_v+1)))
            print('\n')

        if epoch%500 == 0:
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch + epoch_trained}
            torch.save(state, args.model_step_path)
            print("Model saved in epoch:", epoch)

    """train loss in epoches"""
    """total loss (weighted)"""
    x = range(args.train_epoch)

    plt.figure(figsize=(15,5))
    plt.subplot(121)
    plt.grid()
    plt.plot(x, LSTM_Loss_List, '.-')
    plt.xlabel('epoches')
    plt.ylabel('Train loss')

    plt.subplot(122)
    plt.plot(x, Validation_Loss_List, '.-')
    plt.xlabel('epoches')
    plt.ylabel('Validation loss')

    plt.savefig("./image/total_loss_" + args.loss_image_path)
    plt.close()

    """steer loss"""
    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.grid()
    plt.plot(x, steer_loss_list, '.-')
    plt.xlabel('epoches')
    plt.ylabel('Train loss')

    plt.subplot(122)
    plt.plot(x, Vsteer_Loss_List, '.-')
    plt.xlabel('epoches')
    plt.ylabel('Validation loss')

    plt.savefig("./image/steer_loss_" + args.loss_image_path)
    plt.close()

    """speed loss"""
    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.grid()
    plt.plot(x, speed_loss_list, '.-')
    plt.xlabel('epoches')
    plt.ylabel('Train loss')

    plt.subplot(122)
    plt.plot(x, Vspeed_Loss_List, '.-')
    plt.xlabel('epoches')
    plt.ylabel('Validation loss')

    plt.savefig("./image/speed_loss_" + args.loss_image_path)
    plt.close()

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser( description='Self_driving car model parameters with (MESLoss) train with batch 32 and sequence length 16')
    parser.add_argument('--log_path', type=str, default='./data/driving_log.csv', help="log path")
    parser.add_argument('--image_path', type=str, default='./data/IMG/', help="images path")
    parser.add_argument('--image_size', type=set, default=(227, 227), help="image size for training")
    parser.add_argument('--validation_data_size', type=float, default=0.3, help="image size for training")
    parser.add_argument('--GPU_device', type=bool, default=None, help="Use GPU or not")

    parser.add_argument('--lstm_layers_num', type=int, default=2, help="num of lstm")
    parser.add_argument('--lstm_hidden_layers', type=int, default=32, help="num of hidden of lstm")

    parser.add_argument('--batch_size', type=int, default=32, help="batch size for training")
    parser.add_argument('--sequence_length', type=int, default=32, help="batch size for training")
    parser.add_argument('--training_range', type=int, default=None, help="range of data cycle for training")
    parser.add_argument('--validation_range', type=int, default=None, help="range of data cycle for validation")

    parser.add_argument('--lr_rate', type=float, default=1e-5, help="learning rate")
    parser.add_argument('--train_epoch', type=int, default=100, help="training epochs")
    parser.add_argument('--weighted_loss', type=bool, default=True, help="whether weight the loss")
    parser.add_argument('--lambda_value_speed', type=float, default=0.7, help="loss weight for speed loss")
    parser.add_argument('--trained', type=bool, default=False, help="continue train or not")
    parser.add_argument('--speed_data_original', type=bool, default=False, help="normalize the speed data or not")
    parser.add_argument('--loss_function', type=str, default="Mean", help="chose loss function")
    parser.add_argument('--loss_display', type=int, default=700, help="display the loss for every n steps")

    parser.add_argument('--loss_image_path', type=str, default='model_udacity_adc.png', help="loss image saver")
    parser.add_argument('--model_step_path', type=str, default='./models/model_udacity_adc.h5', help="step saver model path")
    parser.add_argument('--model_final_path', type=str, default='./models/model_udacity_adc.pth', help="final model path")

    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
        args.GPU_device = True
    else:
        device = torch.device('cpu')
        args.GPU_device = False
    print("Device running:", device)

    """Load the csv file, read and process the image"""

    train_images, valid_image, train_sequence, valid_sequence, steer_sequence_t, steer_sequence_v = data_loader()

    # transforms
    transformations = transforms.Compose([transforms.ToTensor()])
    # separate and load training data
    training_set = Dataset(train_images, train_sequence, steer_sequence_t, transformations)
    training_generator = data.DataLoader(training_set, shuffle=True, batch_size=args.batch_size, drop_last=True)
    # separate and load validation data
    validation_set = Dataset(valid_image, valid_sequence, steer_sequence_v, transformations)
    validation_generator = data.DataLoader(validation_set, shuffle=True, batch_size=args.batch_size, drop_last=True)

    args.training_range = len(training_generator)
    args.validation_range = len(validation_generator)

    #print(args.validation_range)

    t_iterator = iter(cycle(training_generator))
    v_iterator = iter(cycle(validation_generator))

    # Load model
    # model = CNN()
    # model = CNNLSTM()
    model = CNNLSTMpredict_straight()

    """**loss-function**"""

    criterion_steer = nn.MSELoss()
    criterion_speed = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr_rate)


    model = train(train_iterator=t_iterator, valid_iterator=v_iterator, model=model)

    """3.final saver"""

    state = {
        'model': model.module if device == 'cuda' else model,
    }
    print(state)
    torch.save(state, args.model_final_path, _use_new_zipfile_serialization=False)