import torch
import torch.nn as nn

import io

<<<<<<< HEAD


# class Net(nn.module):
=======
class Net(nn.Module):
  def __init__(self, IN_SIZE, OUT_SIZE, h_size):
    super(Net, self).__init__()

    self.IN_SIZE = IN_SIZE
    self.h_size = h_size
    self.OUT_SIZE = OUT_SIZE

    self.input = nn.Linear(IN_SIZE, h_size)
    self.h0     = nn.Linear(h_size, h_size)
    self.output = nn.Linear(h_size, OUT_SIZE)

  
  def forward(self, x):
    x = torch.tanh(self.input(x))
    x = torch.tanh(self.h0(x))
    x = self.output(x)

    return x


def main():
  IN_SIZE = 2
  OUT_SIZE = 1
  h_size = 2

  # model = Net(IN_SIZE=IN_SIZE, OUT_SIZE=OUT_SIZE, h_size=h_size)


  # torch.jit.load("model.pt")

  model = torch.jit.load('model.pt', map_location=torch.device('cpu'))
  # print(model)
  # input_tensor = torch.rand(1, IN_SIZE)
  # output = model(input_tensor)
  print(model)





if __name__ == "__main__":
  main() 
>>>>>>> b55b8d0da0ac87502abd9cb0f0821a255f0401cb
