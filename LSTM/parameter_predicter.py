import os
import numpy as np

import torch
import torch.nn as nn

from sklearn import preprocessing

from pickle import dump, load

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device is:", device)

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_outputs, dropout=0.3):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm1 = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first = True)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.dout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(hidden_dim, num_outputs)

    def forward(self, x):
        hidden_state = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        cell_state = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)

        out, hidden = self.lstm1(x, (hidden_state, cell_state))
        #out = out.reshape(out.shape[0], -1)
        out = self.fc1(out[:,-1,:])
        out = self.dout(out)
        out = self.fc2(out)
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.num_layers, batch_size, self.hidden_dim).zero_().to(device),
                      weight.new(self.num_layers, batch_size, self.hidden_dim).zero_().to(device))
        return hidden

def load_target():
	data_dir = './'
	f = 'target.npz'
	target = np.load(data_dir + f)['target_spectrum'].astype(np.float32)
	target = torch.from_numpy(target)
	return target

model_file = './model_save.pt'
scaler_file = './parameter_scaler.pkl'
input_dim = 16384
hidden_dim = 256
num_layers = 2
num_outputs = 5

batch_size = 48
sequence_length = 1

if(os.path.exists(model_file) and os.path.exists(scaler_file)):
	model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim,
             num_layers=num_layers, num_outputs=num_outputs)
	model.to(device)
	model.load_state_dict(torch.load(model_file))
	model.eval()
	criterion = torch.nn.MSELoss()
	mm = load(open(scaler_file, 'rb'))

	target = load_target().reshape(-1, sequence_length, input_dim)
	target_params, _ = model(target.to(device))
	print("\nScaled parameters:", target_params[0])
	target_params = mm.inverse_transform(target_params.cpu().detach().numpy())
	print("\nPredicted Parameters for Target:", target_params[0])
else:
	print("Could not find saved model or saved scaler. Please re-train model...")