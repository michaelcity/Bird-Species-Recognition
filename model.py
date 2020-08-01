{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch.nn as nn\n",
    "import torch.nn.functional as F\n",
    "\n",
    "# define the CNN architecture\n",
    "class Net(nn.Module):\n",
    "    def __init__(self):\n",
    "        super(Net, self).__init__()\n",
    "        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)\n",
    "        self.norm2d1 = nn.BatchNorm2d(32)\n",
    "        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)\n",
    "        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)\n",
    "        \n",
    "        # pool\n",
    "        self.pool = nn.MaxPool2d(2, 2)\n",
    "        size_linear_layer = 512\n",
    "        \n",
    "        # linear layer\n",
    "        total_bird_classes = len(CLASSES)\n",
    "        self.fc1 = nn.Linear(128 * 28 * 28, size_linear_layer)\n",
    "        self.fc2 = nn.Linear(size_linear_layer, total_bird_classes)\n",
    "    \n",
    "    def forward(self, x):\n",
    "        super().__init__()\n",
    "        x = self.pool(F.relu(self.norm2d1(self.conv1(x))))\n",
    "        x = self.pool(F.relu(self.conv2(x)))\n",
    "        x = self.pool(F.relu(self.conv3(x)))\n",
    "        \n",
    "        x = x.view(-1, 128 * 28 * 28)\n",
    "        x = F.relu(self.fc1(x))\n",
    "        x = self.fc2(x)\n",
    "        return x\n",
    "\n",
    "# instantiate the CNN\n",
    "model_scratch = Net()\n",
    "\n",
    "# move tensors to GPU if CUDA is available\n",
    "if use_cuda:\n",
    "    model_scratch.cuda()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
