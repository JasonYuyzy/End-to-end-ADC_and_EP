import torch.optim as optim
from torch.utils import data
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import time
from torch.autograd import Variable

from model_three import *
import argparse
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1, 1))


"""**load image data**"""


def augment(imgName, angle):
    name = args.image_path + imgName.split('/')[-1]
    current_image = cv2.imread(name)

    # flip
    if np.random.rand() < 0.5:
        # left-right transfer
        current_image = cv2.flip(current_image, 1)
        angle = angle * -1.0

    # random brightness
    '''current_image = cv2.cvtColor(current_image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    current_image[:, :, 2] = current_image[:, :, 2] * ratio
    current_image = cv2.cvtColor(current_image, cv2.COLOR_HSV2RGB)'''

    # [height, width, deep] crop
    current_image = current_image[50:-23, :, :]  # remove the sky and the car front
    # resize
    current_image = cv2.resize(current_image, args.image_size, cv2.INTER_AREA)
    # rgb2yuv
    #current_image = cv2.cvtColor(current_image, cv2.COLOR_RGB2YUV)

    return current_image, angle

def image_loader(image_pth, angle_list):
    img_list = list()
    steer_list = list()
    """testing"""
    for im_path, steering in zip(image_pth, angle_list):
        # center only
        img, steering = augment(im_path[0], steering)

        img_list.append(img)
        steer_list.append(steering)

    return img_list, steer_list


"""sequence generator"""
def speed_class(speeds):
    speed_classes = []
    pre_speed = speeds[0]
    for i in range(len(speeds)):
        # speed classes, speed states
        if speeds[i] - pre_speed > 0.5:
            speed_classes.append([0, 0, 1])  # speed-up
        elif speeds[i] - pre_speed < -0.14:
            speed_classes.append([1, 0, 0])  # speed-down
        else:
            speed_classes.append([0, 1, 0])  # keeping
        pre_speed = speeds[i]

    a = {'speed_classes': speed_classes}
    speed_generate = pd.DataFrame(a)
    print("Speed classes generated")
    return speed_generate

def speed_sequence(speeds):
    speed_sequence = []
    i_sequence = [speeds[0] for i in range(args.failure_sequence_length)]
    for i in range(len(speeds)):
        # speed sequence 10 speeds
        if i == len(speeds) - 1:
            i_sequence.pop(args.failure_sequence_length-1)
            i_sequence.insert(0, speeds[i])
        else:
            i_sequence.pop(args.failure_sequence_length-1)
            i_sequence.insert(0, speeds[i + 1])
        sq = i_sequence.copy()
        speed_sequence.append(sq)

    a = {'speed_sequence': speed_sequence}
    speed_generate = pd.DataFrame(a)
    print("Speed sequence generated")
    return speed_generate


def steering_generator(steerings):
    steer_sequence = []
    s_sequence = [steerings[0] for i in range(args.failure_sequence_length)]
    for i in range(len(steerings)):
        # steer sequence batch speeds
        if i == len(steerings) - 1:
            s_sequence.pop(args.failure_sequence_length - 1)
            s_sequence.insert(0, steerings[i])
        else:
            s_sequence.pop(args.failure_sequence_length - 1)
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


"""Failure generation"""
def sgn(current_image, steering_sequence, speed_sequence):
    predict_steer_sequence = steering_sequence[0:args.predict_sequence_length]
    prediction_speed_sequence = speed_sequence[0:args.predict_sequence_length]

    image = transformations(current_image)
    image = image.view(1, 3, 227, 227)
    image = Variable(image)

    steer_sqs = torch.Tensor(predict_steer_sequence)
    steer_sqs = steer_sqs.view([1, args.predict_sequence_length, 1])

    speed_sqs = torch.Tensor(prediction_speed_sequence)
    speed_sqs = speed_sqs.view([1, args.predict_sequence_length, 1])

    image, input_steer, input_speed = image.float().to(device), steer_sqs.float().to(device), speed_sqs.float().to(device)

    predict_steer, predict_speed = model_predict(image, input_speed, input_steer)

    if args.GPU_device:
        p_steer = predict_steer.view(-1).cpu().data.numpy()[0]
        p_speed = predict_speed.view(-1).cpu().data.numpy()[0]
    else:
        p_steer = predict_steer.view(-1).data.numpy()[0]
        p_speed = predict_speed.view(-1).data.numpy()[0]


    dif_st = abs(steering_sequence[1] - p_steer) - args.steer_thresholds
    dif_sp = abs(speed_sequence[1] - p_speed) - args.speed_thresholds

    '''print("Steer:")
    print(p_steer)
    print(steering_sequence[1])
    #print("Speed:")
    #print(p_speed)
    #print(speed_sequence[1])
    print('\n')
    time.sleep(2)'''

    if dif_st > 0 or dif_sp > 0:
        return 1
    else:
        return 0

def failure_generation(images, steering_sequence, speed_sequence):
    fail_label_list = []

    number = 0
    for i in range(len(images)):
        count = 0
        for j in range(args.failure_sequence_length):
            if i + j == len(images) - 1:
                break
            count += sgn(images[i+j], steering_sequence[i+j], speed_sequence[i+j])
            if count > 1:
                fail_label_list.append([0, 1]) # not safe
                break
        if count <= 1:
            fail_label_list.append([1, 0]) # safe
            number += 1

    print("The safe prediction number:", number)
    print("Total prediction:", len(fail_label_list))
    b = {'failure_labels': fail_label_list}
    failure_generate = pd.DataFrame(b)

    return failure_generate


"""**data-set collection**"""

scaler.fit_transform(np.array([0,32]).reshape(-1, 1))

def data_loader():
    data_df = pd.read_csv(args.log_path, header=None)
    images = data_df[[0, 1, 2]].values
    speeds = data_df[6].values
    steers = data_df[3].values

    # test sequence
    '''images = images[0:100]
    speeds = speeds[0:100]
    steers = steers[0:100]'''
    #speed_dis(speeds)

    # load image data
    images, steers = image_loader(images, steers)
    print("Image processed")


    # normalize the speed to range -1 ~ 1 from range 0 ~ 32
    if not args.speed_data_original:
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

    # failure label generation
    print("Start generate Failure labels")
    failure_generated = failure_generation(images, steering_sequence, speed_sequences)
    failure_labels = failure_generated['failure_labels'].values
    print("Failure labels generated")


    train_images, valid_images, train_steer_sequence, valid_steer_sequence = train_test_split(images, steering_sequence, test_size=args.validation_data_size, random_state=42, shuffle=False)
    train_speed_sequence, valid_speed_sequence, train_failure_labels, valid_failure_labels = train_test_split(speed_sequences, failure_labels, test_size=args.validation_data_size, random_state=42, shuffle=False)


    return train_images, valid_images, train_speed_sequence, valid_speed_sequence, train_failure_labels, valid_failure_labels, train_steer_sequence, valid_steer_sequence



"""data preparetion"""

class Dataset(data.Dataset):
    def __init__(self, images, speed_sequences, failure_labels, steer_sequence, transforms=None):
        self.images = images
        self.speed_sequences = speed_sequences
        self.failure_labels = failure_labels
        self.steer_sequences = steer_sequence
        self.transform = transforms

    def __getitem__(self, index):
        img = self.images[index]
        batch_speed_sequence = self.speed_sequences[index]
        batch_failure_labels = self.failure_labels[index]
        batch_steer_sequence = self.steer_sequences[index]

        speed_sq = np.array(batch_speed_sequence)
        speed_sq = speed_sq.reshape((args.failure_sequence_length, 1))

        steer_sq = np.array(batch_steer_sequence)
        steer_sq = steer_sq.reshape((args.failure_sequence_length, 1))

        failure_la = np.array(batch_failure_labels)
        failure_la = failure_la.reshape((2))

        img = self.transform(img)

        return (img, steer_sq[1], speed_sq, failure_la, steer_sq)

    # function return the len of the training sets
    def __len__(self):
        return len(self.images)


"""helper function to make getting another batch of data easier"""

def cycle(iterable):
    while True:
        for x in iterable:
            yield x



"""**train function**"""
def train(train_iterator, valid_iterator, model_fail):

    failure_loss_list = list()
    Vfail_Loss_List = list()

    train_accuracy_list = list()
    valid_accuracy_list = list()

    epoch_trained = 0

    if args.trained:
        checkpoint = torch.load(args.model_step_path)
        model_fail.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch_trained = checkpoint['epoch']

    for epoch in range(args.train_epoch):
        model_fail.to(device)
        print("Epoch:", epoch + epoch_trained)

        train_failure_loss = 0.0
        t_num_correct = 0

        model_fail.train()
        for local_batch_t in range(args.training_range):
            imgs, steering_angles, speed_sqs, failure_labs, steer_sqs = next(train_iterator)

            #transfer into torch
            img_inputs, steer_labels, speed_inputs, failure_labels, steer_inputs = imgs.float().to(device), steering_angles.float().to(device), speed_sqs.float().to(device), np.argmax(failure_labs.long(),axis=1).to(device), steer_sqs.float().to(device)

            #optimizer zero gradient
            optimizer.zero_grad()

            model_fail.hidden_cell_sp = (torch.zeros(args.lstm_layers_num, args.batch_size, args.lstm_failure_hidden_layers).to(device), torch.zeros(args.lstm_layers_num, args.batch_size, args.lstm_failure_hidden_layers).to(device))
            model_fail.hidden_cell_st = (torch.zeros(args.lstm_layers_num, args.batch_size, args.lstm_failure_hidden_layers).to(device), torch.zeros(args.lstm_layers_num, args.batch_size, args.lstm_failure_hidden_layers).to(device))


            #training
            output_failures = model_fail(img_inputs, speed_inputs, steer_inputs)

            failure_loss = criterion_failure(output_failures, failure_labels)

            t_pred = output_failures.argmax(dim=1)
            correct_num = torch.eq(t_pred, failure_labels).sum().float().item()
            t_num_correct += correct_num

            loss = failure_loss
            loss.backward()
            optimizer.step()

            train_failure_loss += failure_loss.item()

            if local_batch_t % args.loss_display == 0:
                print('Failure Loss: %.3f '% (failure_loss))
                print('Failure Prediction Accuracy: {:.4%}'.format(correct_num/args.failure_sequence_length))
                print('\n')

        failure_loss_list.append(train_failure_loss/(local_batch_t+1))

        train_accuracy_list.append(t_num_correct/((local_batch_t+1)*args.failure_sequence_length) * 100)

        #validation
        valid_failure_loss = 0.0
        v_num_correct = 0

        model_fail.eval()
        with torch.set_grad_enabled(False):
            for local_batch_v in range(args.validation_range):
                v_imgs, v_steering_angles, v_speed_sqs, v_failure_labs, v_steer_sqs = next(valid_iterator)

                v_img_inputs, v_steer_labels, v_speed_inputs, v_failure_labels, v_steer_inputs = v_imgs.float().to(device), v_steering_angles.float().to(device), v_speed_sqs.float().to(device), np.argmax(failure_labs.long(),axis=1).to(device), v_steer_sqs.float().to(device)

                #optimizer zero gradient
                optimizer.zero_grad()

                # validation
                v_output_failures = model_fail(v_img_inputs, v_speed_inputs, v_steer_inputs)


                va_failure_loss = criterion_failure(v_output_failures, v_failure_labels)

                v_pred = v_output_failures.argmax(dim=1)
                v_correct_num = torch.eq(v_pred, v_failure_labels).sum().float().item()
                v_num_correct += v_correct_num

                valid_failure_loss += va_failure_loss.item()

            Vfail_Loss_List.append(valid_failure_loss/(local_batch_v+1))

            valid_accuracy_list.append(v_num_correct /((local_batch_v+1)*args.failure_sequence_length) * 100)

            print('Failure Valid Loss: %.3f '% (va_failure_loss))
            print('Failure Prediction Valid Accuracy: {:.4%}'.format(v_correct_num/args.failure_sequence_length))
            print('\n')

        if epoch%500 == 0:
            state = {'model': model_fail.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch + epoch_trained}
            torch.save(state, args.model_step_path)
            print("Model saved in epoch:", epoch)

    """train loss in epoches"""
    """failure loss"""
    x = range(args.train_epoch)

    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.grid()
    plt.plot(x, failure_loss_list, '.-')
    plt.xlabel('epoches')
    plt.ylabel('Train loss')

    plt.subplot(122)
    plt.grid()
    plt.plot(x, Vfail_Loss_List, '.-')
    plt.xlabel('epoches')
    plt.ylabel('Validation loss')

    plt.savefig("./image/failure_loss_" +args.loss_image_path)
    plt.close()

    """accuracy"""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    lns1 = ax.plot(x, train_accuracy_list, '-', label='train accuracy')
    lns2 = ax.plot(x, valid_accuracy_list, '-', label='validation accuracy')
    # added these two lines
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0)
    ax.grid()
    ax.set_xlabel("Epochs")
    ax.set_ylabel("accuracy rate %")
    ax.set_xlim(0, args.train_epoch + 3)
    ax.set_ylim(0, 110)

    plt.savefig("./image/failure_accuracy_" + args.loss_image_path)
    plt.close()

    return model_fail


if __name__ == "__main__":
    parser = argparse.ArgumentParser( description='Self_driving car model parameters with (MESLoss) train with batch 32 and sequence length 64')
    parser.add_argument('--log_path', type=str, default='./data/driving_log.csv', help="log path")
    parser.add_argument('--image_path', type=str, default='./data/IMG/', help="images path")
    parser.add_argument('--image_size', type=set, default=(227, 227), help="image size for training")
    parser.add_argument('--validation_data_size', type=float, default=0.3, help="image size for training")
    parser.add_argument('--model_predict', type=str, default='./models/model_udacity_adc.pth', help="trained model for steering and speed prediction")
    parser.add_argument('--GPU_device', type=bool, default=None, help="Use GPU or not")

    parser.add_argument('--steer_thresholds', type=float, default=0.2, help="thresholds defining steering angle")
    parser.add_argument('--speed_thresholds', type=float, default=0.7, help="thresholds defining speed")

    parser.add_argument('--lstm_layers_num', type=int, default=2, help="num of lstm")
    parser.add_argument('--lstm_predict_hidden_layers', type=int, default=32, help="num of hidden of lstm predict")
    parser.add_argument('--lstm_failure_hidden_layers', type=int, default=64, help="num of hidden of lstm")

    parser.add_argument('--batch_size', type=int, default=64, help="batch size for training")
    parser.add_argument('--batch_size_prediction', type=int, default=1, help="batch size for training")
    parser.add_argument('--predict_sequence_length', type=int, default=32, help="batch size for training")
    parser.add_argument('--failure_sequence_length', type=int, default=64, help="batch size for training")
    parser.add_argument('--training_range', type=int, default=None, help="range of data cycle for training")
    parser.add_argument('--validation_range', type=int, default=None, help="range of data cycle for validation")
    parser.add_argument('--speed_data_original', type=bool, default=False, help="normalize the speed data or not")

    parser.add_argument('--lr_rate', type=float, default=1e-5, help="learning rate")
    parser.add_argument('--train_epoch', type=int, default=1000, help="training epochs")
    parser.add_argument('--trained', type=bool, default=False, help="continue train or not")
    parser.add_argument('--loss_function', type=str, default="Mean", help="chose loss function")
    parser.add_argument('--loss_display', type=int, default=700, help="display the loss for every n steps")

    parser.add_argument('--loss_image_path', type=str, default='model_udacity_ep.png', help="loss image saver")
    parser.add_argument('--model_step_path', type=str, default='./models/model_udacity_ep.h5', help="step saver model path")
    parser.add_argument('--model_final_path', type=str, default='./models/model_udacity_ep.pth', help="final model path")

    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
        args.GPU_device = True
    else:
        device = torch.device('cpu')
        args.GPU_device = False
    print("Device running:", device)

    # load the trained model
    checkpoint = torch.load(args.model_predict, map_location=device)
    model_predict = checkpoint['model']
    model_predict.eval()
    model_predict.hidden_cell_sp = (torch.zeros(args.lstm_layers_num, args.batch_size_prediction, args.lstm_predict_hidden_layers).to(device), torch.zeros(args.lstm_layers_num, args.batch_size_prediction, args.lstm_predict_hidden_layers).to(device))
    model_predict.hidden_cell_st = (torch.zeros(args.lstm_layers_num, args.batch_size_prediction, args.lstm_predict_hidden_layers).to(device), torch.zeros(args.lstm_layers_num, args.batch_size_prediction, args.lstm_predict_hidden_layers).to(device))
    print(model_predict)

    # transforms
    transformations = transforms.Compose([transforms.ToTensor()])

    """Load the csv file, read and process the image"""

    train_images, valid_image, train_sequence, valid_sequence, train_class, valid_class, steer_sequence_t, steer_sequence_v = data_loader()

    # separate and load training data
    training_set = Dataset(train_images, train_sequence, train_class, steer_sequence_t, transformations)
    training_generator = data.DataLoader(training_set, shuffle=True, batch_size=args.batch_size, drop_last=True)
    # separate and load validation data
    validation_set = Dataset(valid_image, valid_sequence, valid_class, steer_sequence_v, transformations)
    validation_generator = data.DataLoader(validation_set, shuffle=True, batch_size=args.batch_size, drop_last=True)

    args.training_range = len(training_generator)
    args.validation_range = len(validation_generator)

    #print(args.validation_range)

    t_iterator = iter(cycle(training_generator))
    v_iterator = iter(cycle(validation_generator))

    # Load model
    # model = CNN()
    # model = CNNLSTM()
    model_fail = CNNLSTMfail()

    """**loss-function**"""
    criterion_failure = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_fail.parameters(), lr=args.lr_rate)

    # start training
    model_fail = train(train_iterator=t_iterator, valid_iterator=v_iterator, model_fail=model_fail)


    """3.final saver"""

    state = {
        'model': model_fail.module if device == 'cuda' else model_fail,
    }
    print(state)
    torch.save(state, args.model_final_path, _use_new_zipfile_serialization=False)