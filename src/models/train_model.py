import torch
import torch.nn.functional as F
from torch import nn, optim
import click

import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], ".."))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from data.make_dataset import MNISTdata

print("Working directory : {}".format(os.getcwd()))

class MyAwesomeModel(nn.Module):
    def __init__(self, hidden_size: int, output_size: int, drop_p: float = 0.3) -> None:
        """Builds a feedforward network with arbitrary hidden layers.
        Arguments
        ---------
        hidden_size: integer, size of dense layer
        output_size: number of classes
        drop_p: dropout rate
        """
        super().__init__()
        # Input to a hidden layer
        self.num_classes = output_size

        self.arch = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=16, kernel_size=3, padding=1, stride=1
            ),
            # convolution output dim (16, 28, 28)
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # pooling output dim (16, 14, 14)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=5, padding=2),
            nn.Dropout2d(p=drop_p),
            # convolution output dim (8, 14, 14)
            nn.MaxPool2d(kernel_size=2, stride=2),
            # polling output dim (8, 7, 7)
            nn.ReLU(inplace=True),
        )

        # fully connected output layers
        # [(W−K+2P)/S]+1
        self.fc1_features = 8 * 7 * 7
        self.fc1 = nn.Linear(in_features=self.fc1_features, out_features=hidden_size)
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=self.num_classes)

    def forward(self, x):

        x = self.arch(x)
        x = x.view(-1, self.fc1_features)
        x = F.relu(self.fc1(x))

        return F.log_softmax(self.fc2(x), dim=1)


def validation(model, testloader, criterion):
    accuracy = 0
    test_loss = 0
    for images, labels in testloader:

        # images = images.resize_(images.size()[0], 784)

        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ## Calculating the accuracy
        # Model's output is log-softmax, take exponential to get the probabilities
        ps = torch.exp(output)
        # Class with highest probability is our predicted class, compare with true label
        equality = labels.data == ps.max(1)[1]
        # Accuracy is number of correct predictions divided by all predictions, just take the mean
        accuracy += equality.type_as(torch.FloatTensor()).mean()

    return test_loss, accuracy


def model_train(
    model, trainloader, testloader, criterion, optimizer=None, epochs=5, print_every=40
):

    if optimizer is None:
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    steps = 0
    running_loss = 0
    for e in range(epochs):
        # Model in training mode, dropout is on
        model.train()
        for images, labels in trainloader:
            steps += 1

            # Flatten images into a 784 long vector
            # images.resize_(images.size()[0], 784)

            optimizer.zero_grad()

            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                # Model in inference mode, dropout is off
                model.eval()

                # Turn off gradients for validation, will speed up inference
                with torch.no_grad():
                    test_loss, accuracy = validation(model, testloader, criterion)

                print(
                    "Epoch: {}/{}.. ".format(e + 1, epochs),
                    "Training Loss: {:.3f}.. ".format(running_loss / print_every),
                    "Test Loss: {:.3f}.. ".format(test_loss / len(testloader)),
                    "Test Accuracy: {:.3f}".format(accuracy / len(testloader)),
                )

                running_loss = 0

                # Make sure dropout and grads are on for training
                model.train()


@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
def train(lr):
    print("Training day and night")
    print(lr)
    model1 = MyAwesomeModel(784, 10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model1.parameters(), lr=lr)

    train_data = torch.load("../data/processed/train.pth")
    test_data = torch.load("../data/processed/test.pth")

    trainloader = torch.utils.data.DataLoader(
        train_data, batch_size=64, shuffle=True
    )
    testloader = torch.utils.data.DataLoader(
        test_data, batch_size=64, shuffle=True
    )

    model_train(
        model1, trainloader, testloader, criterion, optimizer, epochs=5, print_every=40
    )

    checkpoint = {
        "hidden_size": 784,
        "output_size": 10,
        "state_dict": model1.state_dict(),
    }

    torch.save(checkpoint, f"C:/Users/zuzal/Masters/MLOPs/MLOPS_S2/models/checkpoint.pth")


cli.add_command(train)


if __name__ == "__main__":
    cli()