import argparse
import base64
from datetime import datetime
import os
import shutil
import cv2

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO

# from keras.models import load_model
# import h5py
# from keras import __version__ as keras_version
from torch.autograd import Variable
import torchvision.transforms as transforms
from model_three import *
from sklearn.preprocessing import MinMaxScaler

speed_scaler = MinMaxScaler(feature_range=(-1, 1))
steer_scaler = MinMaxScaler(feature_range=(-1, 1))

import time
import random


sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

transformations = transforms.Compose([transforms.ToTensor()])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device running:", device)


class SimplePIController:
    def __init__(self, Kp, Ki):
        self.Kp = Kp
        self.Ki = Ki
        self.set_point = 0.
        self.error = 0.
        self.integral = 0.

    def set_desired(self, desired):
        self.set_point = desired

    def update(self, measurement):
        # proportional error
        self.error = self.set_point - measurement

        # integral error
        self.integral += self.error

        return self.Kp * self.error + self.Ki * self.integral



def speed_to_throttle(p_speed, c_speed):
    if p_speed - c_speed > 0.5:
        return 0.7
    elif p_speed - c_speed < -0.14:
        return -0.9
    else:
        return 0.07

def img_process(image_array):

    # locate the main part of the image
    current_image = image_array[50:-23, :, :]
    # resize
    current_image = cv2.resize(current_image, args.image_size, cv2.INTER_AREA)
    # RGB to YUV
    current_image = cv2.cvtColor(current_image, cv2.COLOR_RGB2YUV)

    image = transformations(current_image)
    #print("SIZEEEEEE:", image.size())
    image = image.view(1, 3, 227, 227)
    #print("SIZEEEEEE:", image.size())
    image = Variable(image)

    return image



@sio.on('telemetry')
def telemetry(sid, data):
    #print('Running')
    if data:
        # The current steering angle of the car
        steering_angle = float(data["steering_angle"])
        # The current speed of the car
        speed = float(data["speed"])
        # The current image from the center camera of the car
        imgString = data["image"]
        image = Image.open(BytesIO(base64.b64decode(imgString)))
        image_array = np.asarray(image)

        # generate the speed sequence
        speed_sequence.pop(args.sequence_length - 1)
        speed_sequence.insert(0, speed)

        steer_sequence.pop(args.sequence_length - 1)
        steer_sequence.insert(0, steering_angle)

        input_speed = np.append(np.array(speed_sequence.copy()), [0, 32])
        input_speed = speed_scaler.fit_transform(input_speed.reshape(-1, 1))
        input_speed = np.delete(input_speed, [len(input_speed) - 2, len(input_speed) - 1])
        '''
        input_speed = speed_sequence.copy()'''

        input_steer = np.append(np.array(steer_sequence.copy()), [-25, 25])
        input_steer = steer_scaler.fit_transform(input_steer.reshape(-1, 1))
        input_steer = np.delete(input_steer, [len(input_steer) - 2, len(input_steer) - 1])


        # generate the steer sequence
        input_steer = torch.Tensor(input_steer)
        #input_steer = torch.Tensor(steer_sequence.copy())
        input_speed = torch.Tensor(input_speed)

        # preprocess image input
        image = img_process(image_array)
        input_speed = input_speed.view([1, args.sequence_length, 1])
        input_steer = input_steer.view([1, args.sequence_length, 1])


        # speed straight
        steering_angle, p_speed = model(image, input_speed, input_steer)#.view(-1).data.numpy()[0]
        steering_angle = steering_angle.view(-1).data.numpy()[0]
        #print(p_speed)
        p_speed = p_speed.view(-1).data.numpy()
        p_speed = speed_scaler.inverse_transform(p_speed.reshape(-1, 1))[0]
        print("speed predict", p_speed[0])

        print("Steering angle: ", steering_angle)


        if speed < 11:
            throttle = 0.5
        elif speed > 20:
            throttle = -0.9
        else:
            throttle = speed_to_throttle(p_speed[0], speed)
        send_control(steering_angle, throttle)


        # save frame
        '''if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))'''
    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    parser.add_argument('--lstm_layers_num', type=int, default=2, help="num of lstm")
    parser.add_argument('--lstm_hidden_layers', type=int, default=32, help="num of hidden of lstm")

    parser.add_argument('--batch_size', type=int, default=1, help="batch size for testing")
    parser.add_argument('--sequence_length', type=int, default=32, help="speed sequence length")
    parser.add_argument('--image_size', type=set, default=(227, 227), help="image size for training")
    args = parser.parse_args()

    speed_sequence = [0.0 for i in range(args.sequence_length)]
    steer_sequence = [0.0 for i in range(args.sequence_length)]

    # check that model Keras version is same as local Keras version
    #change
    checkpoint = torch.load(args.model, map_location=device)
    model = checkpoint['model']
    model.eval()
    print(model)

    model.hidden_cell_sp = (torch.zeros(args.lstm_layers_num, args.batch_size, args.lstm_hidden_layers).to(device), torch.zeros(args.lstm_layers_num, args.batch_size, args.lstm_hidden_layers).to(device))
    model.hidden_cell_st = (torch.zeros(args.lstm_layers_num, args.batch_size, args.lstm_hidden_layers).to(device), torch.zeros(args.lstm_layers_num, args.batch_size, args.lstm_hidden_layers).to(device))

    #model.eval()
    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
