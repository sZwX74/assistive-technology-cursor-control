import torch
import pytorch_model_class
from pytorch_model_class import DEVICE
import matplotlib.pylab as plt
import data_loader
from tqdm import tqdm

def train(model, epochs = 10, learning_rate = 0.05,
          train_batch_size = 2000, validation_batch_size = 2000,
          save_on_epochs = False,
          train_noise = False, validation_noise = False):
    # load data
    train_data, validation_data = data_loader.load_mnist_data(train_noise = train_noise, validation_noise = validation_noise)
    if train_batch_size == -1:
        train_batch_size = len(train_data)
    if validation_batch_size == -1:
        validation_batch_size = len(validation_data)
    train_loader, validation_loader = data_loader.create_data_loader(train_data, validation_data,\
                                                                     train_batch_size = train_batch_size,
                                                                     validation_batch_size = validation_batch_size)

    # Create the criterion function
    criterion = torch.nn.CrossEntropyLoss()

    # create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    useful_stuff = {'training_loss': [], 'train_accuracy': [], 'validation_accuracy': []}

    for epoch in range(epochs):
        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            y = y.to(DEVICE)
            z = model(x.view(-1, 28 * 28).to(DEVICE))
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()
            useful_stuff['training_loss'].append(loss.data.item())

        # train accuracy
        train_correct = 0
        for x, y in tqdm(train_loader):
            y = y.to(DEVICE)
            z = model(x.view(-1, 28 * 28).to(DEVICE))
            _, label = torch.max(z, 1)
            train_correct += (label == y).sum().item()

        train_accuracy = 100 * (train_correct / len(train_data))
        useful_stuff['train_accuracy'].append(train_accuracy)

        # validation accuracy
        validation_correct = 0
        for x, y in tqdm(validation_loader):
            y = y.to(DEVICE)
            z = model(x.view(-1, 28 * 28).to(DEVICE))
            _, label = torch.max(z, 1)
            validation_correct += (label == y).sum().item()

        validation_accuracy = 100 * (validation_correct / len(validation_data))
        useful_stuff['validation_accuracy'].append(validation_accuracy)

        print('trained epoch: ', epoch, ' train_accuracy: ', train_accuracy, ' validation_accuracy: ', validation_accuracy)
        if save_on_epochs:
            model.save_model(path='./saved_models')  # to save every stage of model

    return useful_stuff


if __name__ == "__main__":
    # Set the parameters and create the model
    D_in = 28 * 28
    H1 = 128
    H2 = 128
    D_out = 10

    model = pytorch_model_class.NetReluShallow(D_in, H1, H2, D_out).to(DEVICE)
    # model = pytorch_model_class.NetReluDeep()

    # train and save model

    # uncomment this line if you want to continue training an existing model
    # model.load_model(path = './saved_models')
    training_results = train(model,
                             epochs = 50,
                             learning_rate = 0.005,
                             train_batch_size = 2000,
                             validation_batch_size = 2000,
                             save_on_epochs = True,
                             train_noise = False,
                             validation_noise = False)
    model.save_model(path = './saved_models')

    # plot the training curves
    plt.plot(training_results['training_loss'], label='NetRelu')
    plt.ylabel('loss')
    plt.title('training loss iterations')
    plt.legend()
    plt.show()

    plt.plot(training_results['train_accuracy'], label='train_accuracy')
    plt.plot(training_results['validation_accuracy'], label='validation_accuracy')
    plt.ylabel('train vs validation accuracy')
    plt.xlabel('Iteration')
    plt.legend()
    plt.show()

    # print the final accuracy
    print('final training_loss: ', training_results['training_loss'][-1])
    print('final train_accuracy: ', training_results['train_accuracy'][-1])
    print('final validation_accuracy: ', training_results['validation_accuracy'][-1])