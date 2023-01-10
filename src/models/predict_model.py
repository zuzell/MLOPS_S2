import torch
from torch import nn
from train_model import MyAwesomeModel

import os
import sys


sys.path.insert(1, os.path.join(sys.path[0], ".."))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from data.make_dataset import MNISTdata

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

def evaluate(model_checkpoint):

    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)


    test_data = torch.load("data/processed/test.pth")


    testloader = torch.utils.data.DataLoader(
        test_data, batch_size=64, shuffle=True
    )


    # TODO: Implement evaluation logic here
    checkpoint = torch.load(model_checkpoint)
    model1 = MyAwesomeModel(checkpoint["hidden_size"], checkpoint["output_size"])
    model1.load_state_dict(checkpoint["state_dict"])

    criterion = nn.CrossEntropyLoss()

    test_loss, accuracy = validation(model1, testloader, criterion)
    print(f"test loss: {test_loss}, accuracy: {accuracy}")


if __name__ == "__main__":
    evaluate("C:/Users/zuzal/Masters/MLOPs/MLOPS_S2/models/checkpoint.pth")