import torch
import torch.nn as nn

class single(nn.Module):
    def __init__(self):
        super(single, self).__init__()
        # CNN convolutional layers
        self.cnn_img = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4),  # cnn1
            nn.ReLU(),
            nn.BatchNorm2d(num_features=96),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, stride=1),  # cnn2
            nn.ReLU(),
            nn.BatchNorm2d(num_features=256),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, stride=1),  # cnn3
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, stride=1),  # cnn4
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, stride=1),  # cnn5
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, stride=1),  # cnn6
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, stride=1),  # cnn7
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc_img = nn.Sequential(
            nn.Linear(in_features=256 * 1, out_features=1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )

    def forward(self, input_img):
        # CNN
        input_img = input_img.view(input_img.size(0), 3, 227, 227)
        img_out = self.cnn_img(input_img)
        img_out = img_out.view(img_out.size(0), -1)  # resize
        steer_out = self.fc_img(img_out)
        return steer_out

class CNNLSTMpredict_straight(nn.Module):
    def __init__(self):
        super(CNNLSTMpredict_straight, self).__init__()
        # CNN convolutional layers
        self.cnn_img = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4),  # cnn1
            nn.ReLU(),
            nn.BatchNorm2d(num_features=96),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, stride=1),  # cnn2
            nn.ReLU(),
            nn.BatchNorm2d(num_features=256),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, stride=1),  # cnn3
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, stride=1),  # cnn4
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, stride=1),  # cnn5
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, stride=1),  # cnn6
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, stride=1),  # cnn7
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc_img = nn.Sequential(
            nn.Linear(in_features=256 * 1, out_features=1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU()
        )

        # LSTM_speed
        self.lstm_sp = nn.LSTM(input_size=1, hidden_size=32, num_layers=2, batch_first=True)
        self.hidden_cell_sp = None

        #LSTM_steer
        self.lstm_st = nn.LSTM(input_size=1, hidden_size=32, num_layers=2, batch_first=True)
        self.hidden_cell_st = None

        # LSTM linear Fully Connection for speed
        self.fc_lstm_sp = nn.Sequential(
            nn.Linear(in_features=32 * 32, out_features=1024),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ELU(),
            nn.Dropout(0.5)
        )

        # LSTM linear Fully Connection for steer
        self.fc_lstm_st = nn.Sequential(
            nn.Linear(in_features=32 * 32, out_features=1024),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ELU(),
            nn.Dropout(0.5)
        )

        # concatenate speed
        self.fc_merge_sp = nn.Sequential(
            nn.Linear(1536, 512),
            nn.ELU(),
            nn.Linear(512, 1)
        )

        # concatenate steer
        self.fc_merge_st = nn.Sequential(
            nn.Linear(1536, 512),
            nn.ELU(),
            nn.Linear(512, 1)
        )

    def forward(self, input_img, input_speed, input_steer):
        # CNN
        input_img = input_img.view(input_img.size(0), 3, 227, 227)
        img_out = self.cnn_img(input_img)
        img_out = img_out.view(img_out.size(0), -1)  # resize
        img_out = self.fc_img(img_out)

        # LSTM_speed
        input_speed = input_speed.view(input_speed.size(0), 32, -1)
        out_speed, self.hidden_cell_sp = self.lstm_sp(input_speed, self.hidden_cell_sp)
        speed_out = out_speed.reshape(out_speed.size(0), -1)  # resize
        speed_out = self.fc_lstm_sp(speed_out)

        #LSTM_steer
        input_steer = input_steer.view(input_steer.size(0), 32, -1)
        out_steer, self.hidden_cell_st = self.lstm_st(input_steer, self.hidden_cell_st)
        steer_out = out_steer.reshape(out_steer.size(0), -1)  # resize
        steer_out = self.fc_lstm_st(steer_out)

        # Concatenate speed
        speed_merge = torch.cat((speed_out, img_out), dim=1)
        speed_merge = torch.cat((speed_merge, steer_out), dim=1)
        speed_merge_out = self.fc_merge_sp(speed_merge)

        # Concatenate steer
        steer_merge = torch.cat((img_out, steer_out), dim=1)
        steer_merge = torch.cat((steer_merge, speed_out), dim=1)
        steer_merge_out = self.fc_merge_st(steer_merge)

        return steer_merge_out, speed_merge_out


class CNNLSTMpredict_img(nn.Module):
    def __init__(self):
        super(CNNLSTMpredict_img, self).__init__()
        # CNN convolutional layers
        self.cnn_img = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4),  # cnn1
            nn.ReLU(),
            nn.BatchNorm2d(num_features=96),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, stride=1),  # cnn2
            nn.ReLU(),
            nn.BatchNorm2d(num_features=256),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, stride=1),  # cnn3
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, stride=1),  # cnn4
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, stride=1),  # cnn5
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, stride=1),  # cnn6
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, stride=1),  # cnn7
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc_img = nn.Sequential(
            nn.Linear(in_features=256 * 1, out_features=1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU()
        )

        # LSTM_speed
        self.lstm_sp = nn.LSTM(input_size=1, hidden_size=32, num_layers=2, batch_first=True)
        self.hidden_cell_sp = None

        #LSTM_steer
        self.lstm_st = nn.LSTM(input_size=1, hidden_size=32, num_layers=2, batch_first=True)
        self.hidden_cell_st = None

        # LSTM linear Fully Connection for speed
        self.fc_lstm_sp = nn.Sequential(
            nn.Linear(in_features=32 * 32, out_features=1024),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ELU(),
            nn.Dropout(0.5)
        )

        # LSTM linear Fully Connection for steer
        self.fc_lstm_st = nn.Sequential(
            nn.Linear(in_features=32 * 32, out_features=1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # concatenate speed
        self.fc_merge_sp = nn.Sequential(
            nn.Linear(1536, 512),
            nn.ELU(),
            nn.Linear(512, 1)
        )

        # FC steer
        self.fc_final_steer = nn.Sequential(
            nn.Linear(512, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )

    def forward(self, input_img, input_speed, input_steer):
        # CNN
        input_img = input_img.view(input_img.size(0), 3, 227, 227)
        img_out = self.cnn_img(input_img)
        img_out = img_out.view(img_out.size(0), -1)  # resize
        img_out = self.fc_img(img_out)

        # LSTM_speed
        input_speed = input_speed.view(input_speed.size(0), 32, -1)
        out_speed, self.hidden_cell_sp = self.lstm_sp(input_speed, self.hidden_cell_sp)
        speed_out = out_speed.reshape(out_speed.size(0), -1)  # resize
        speed_out = self.fc_lstm_sp(speed_out)

        #LSTM_steer
        input_steer = input_steer.view(input_steer.size(0), 32, -1)
        out_steer, self.hidden_cell_st = self.lstm_st(input_steer, self.hidden_cell_st)
        steer_out = out_steer.reshape(out_steer.size(0), -1)  # resize
        steer_out = self.fc_lstm_st(steer_out)

        # Concatenate speed
        speed_merge = torch.cat((speed_out, img_out), dim=1)
        speed_merge = torch.cat((speed_merge, steer_out), dim=1)
        speed_merge_out = self.fc_merge_sp(speed_merge)

        # FC steer
        steer_out = self.fc_final_steer(img_out)

        return steer_out, speed_merge_out


class CNNLSTMfail(nn.Module):
    def __init__(self):
        super(CNNLSTMfail, self).__init__()
        # CNN convolutional layers
        self.cnn_img = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4),  # cnn1
            nn.ReLU(),
            nn.BatchNorm2d(num_features=96),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, stride=1),  # cnn2
            nn.ReLU(),
            nn.BatchNorm2d(num_features=256),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, stride=1),  # cnn3
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, stride=1),  # cnn4
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, stride=1),  # cnn5
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, stride=1),  # cnn6
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, stride=1),  # cnn7
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc_img = nn.Sequential(
            nn.Linear(in_features=256 * 1, out_features=1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU()
        )

        # LSTM_speed
        self.lstm_sp = nn.LSTM(input_size=1, hidden_size=64, num_layers=2, batch_first=True)
        self.hidden_cell_sp = None

        #LSTM_steer
        self.lstm_st = nn.LSTM(input_size=1, hidden_size=64, num_layers=2, batch_first=True)
        self.hidden_cell_st = None

        # LSTM linear Fully Connection
        self.fc_lstm_st = nn.Sequential(
            nn.Linear(in_features=64 * 64, out_features=1024),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ELU(),
            nn.Dropout(0.5)
        )

        self.fc_lstm_sp = nn.Sequential(
            nn.Linear(in_features=64 * 64, out_features=1024),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ELU(),
            nn.Dropout(0.5)
        )

        # concatenate failure
        self.fc_merge_failure = nn.Sequential(
            nn.Linear(1536, 512),
            nn.ELU(),
            nn.Linear(512, 2),
            nn.Softmax(1)
        )



    def forward(self, input_img, input_speed, input_steer):
        # CNN
        input_img = input_img.view(input_img.size(0), 3, 227, 227)
        img_out = self.cnn_img(input_img)
        img_out = img_out.view(img_out.size(0), -1)  # resize
        img_out = self.fc_img(img_out)

        # LSTM_speed
        input_speed = input_speed.view(input_speed.size(0), 64, -1)
        out_speed, self.hidden_cell_sp = self.lstm_sp(input_speed, self.hidden_cell_sp)
        speed_out = out_speed.reshape(out_speed.size(0), -1)  # resize
        speed_out = self.fc_lstm_sp(speed_out)

        #LSTM_steer
        input_steer = input_steer.view(input_steer.size(0), 64, -1)
        out_steer, self.hidden_cell_st = self.lstm_st(input_steer, self.hidden_cell_st)
        steer_out = out_steer.reshape(out_steer.size(0), -1)  # resize
        steer_out = self.fc_lstm_st(steer_out)

        # Concatenate failure
        failure_merge = torch.cat((img_out, speed_out), dim=1)
        failure_merge = torch.cat((failure_merge, steer_out), dim=1)
        failure_merge_out = self.fc_merge_failure(failure_merge)

        return failure_merge_out