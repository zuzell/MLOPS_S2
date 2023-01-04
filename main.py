import argparse
import sys
import torchvision
import torch
import click
from torch import optim, nn

from data import mnist
from model import MyAwesomeModel, train, validation
import model


@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-3, help='learning rate to use for training')
def train(lr):
    print("Training day and night")
    print(lr)

    
    model1 = MyAwesomeModel(784, 10, [512, 256, 128])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model1.parameters(), lr=lr)
    trainloader, testloader = mnist()

    model.train(model1, trainloader, testloader, criterion, optimizer, epochs=5, print_every=40)

    checkpoint = {'hidden_size': 128,
            'output_size': 10,
            'state_dict': model1.state_dict()}

    torch.save(checkpoint, 'checkpoint.pth')




@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):

    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model1 = torch.load(model_checkpoint)
    _, test_set = mnist()

    _, testloader = mnist()
    criterion = nn.NLLLoss()

    test_loss, accuracy = validation(model1, testloader, criterion)
    print(f'test loss: {test_loss}, accuracy: {accuracy}')

cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()


    
    
    
    