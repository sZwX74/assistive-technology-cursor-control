import torch
import pytorch_model_class
import matplotlib.pylab as plt
import data_loader

def eval(model, train_noise = False, validation_noise = False):
    # load data
    train_data, validation_data = data_loader.load_mnist_data(train_noise = train_noise, validation_noise = validation_noise)
    train_loader, validation_loader = data_loader.create_data_loader(train_data, validation_data)

    # train accuracy
    train_correct = 0
    for x, y in train_loader:
        z = model(x.view(-1, 28 * 28))
        _, label = torch.max(z, 1)
        train_correct += (label == y).sum().item()

    train_accuracy = 100 * (train_correct / len(train_data))

    # validation accuracy
    validation_correct = 0
    for x, y in validation_loader:
        z = model(x.view(-1, 28 * 28))
        _, label = torch.max(z, 1)
        validation_correct += (label == y).sum().item()

    validation_accuracy = 100 * (validation_correct / len(validation_data))


    return train_accuracy, validation_accuracy


if __name__ == "__main__":
    # Set the parameters and create the model
    D_in = 28 * 28
    H1 = 100
    H2 = 100
    D_out = 10

    model = pytorch_model_class.NetReluShallow(D_in, H1, H2, D_out)
    # model = pytorch_model_class.NetReluDeep()

    # load model and evalulate
    model.load_model(path = './saved_models')
    train_accuracy, validation_accuracy = eval(model, train_noise = False, validation_noise = False)

    # print the final accuracy
    print('train_accuracy: ', train_accuracy)
    print('validation_accuracy: ', validation_accuracy)