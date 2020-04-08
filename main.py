from ml.models import model
import os
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from ml.dataset import EEGDataset
import numpy as np
from sklearn.model_selection import ParameterGrid
import cv2
from skimage.io import imread
from scipy import ndimage as ndi

dimension = 0
data = []
labels = []
data_dir = 'ml/datasets'
saved_model_dir = 'ml/saved_model/'
input = 'Kevin_8.5_panda_inv_resized_2_0_2650.csv'
bitmapping_threshold = 56
mask_threshold = 10

BATCH_SIZE = 1
MAX_EPOCHS = 200
LEARNING_RATE = 0.0000001
MOMENTUM = 0.85
WEIGHT_DECAY = 0.0
TRAIN_VAL_RATIO = 0.8


def load_data(data_dir):
    for file in os.listdir(data_dir):
        if file == input:
            continue
        csv_data = np.genfromtxt(f"{data_dir}/{file}", delimiter=',')
        data.extend(csv_data[:, :-1])
        labels.extend(csv_data[:, -1])


def generate_prediction(filename, nn, shape, threshold=False, denoise=False):
    csv_data = np.genfromtxt(f"{data_dir}/{filename}", delimiter=',')
    X = csv_data[:, :-1]
    pred_y = nn(Variable(torch.from_numpy(X))).data.numpy()
    if denoise:
        pred_y -= 0.5

        for row in pred_y:
            for i in range(len(row)):
                if row[i] < 2e+01:
                    row[i] = 0
                if row[i] >= 2e+01:
                    row[i] = row[i] * 2

    np.savetxt("ml/bitmapping.csv", pred_y)
    if threshold:
        processed_y = np.zeros(pred_y.shape)
        for i in range(len(pred_y)):
            if pred_y[i] > threshold:
                processed_y[i] = 255
            else:
                processed_y[i] = 0
        processed_y = np.reshape(processed_y, shape)
        cv2.imwrite("ml/prediction_Kevin.png", processed_y)
    else:
        pred_y = np.reshape(pred_y, shape)
        cv2.imwrite("ml/prediction_Kevin.png", pred_y)


def weights_init_uniform(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # apply a uniform distribution to the weights and a bias=0
        y = 1/np.sqrt(dimension)
        m.weight.data.uniform_(-1, 1)
        m.bias.data.fill_(0)


# Create training and validation split and load into DataLoader
def split_load_data(data, labels):
    # Shuffle data
    perm = np.random.permutation(len(data))
    data = np.asarray(data)[perm]
    labels = np.asarray(labels)[perm]

    split = int(TRAIN_VAL_RATIO * len(data))
    train_x, train_y = data[0:split], labels[0:split]
    val_x, val_y = data[split:], labels[split:]
    train_set = EEGDataset(train_x, train_y)
    val_set = EEGDataset(val_x, val_y)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
    return train_loader, val_loader


def grid_search(nn, train_loader, val_loader):
    # Perform grid search for turning hyperparameters
    param_grid = {'BATCH_SIZE': [50], 'MAX_EPOCHS': [500],
                  'LEARNING_RATE': [0.0000001],
                  'MOMENTUM': [0.8, 0.85, 0.9], 'WEIGHT_DECAY': [0]}

    grid = ParameterGrid(param_grid)
    for params in grid:

        # Initialize weights uniformly from -1 to 1
        nn.apply(weights_init_uniform)

        nn.train1(train_loader, val_loader, lr=params['LEARNING_RATE'], weight_decay=params['WEIGHT_DECAY'],
                  momentum=params['MOMENTUM'], max_epochs=params['MAX_EPOCHS'], batch_size=params['BATCH_SIZE'])


def train_save(nn, train_loader, val_loader):
    nn.train1(train_loader, val_loader, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
              momentum=MOMENTUM, max_epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, plot=True, save=False)
    if not os.path.exists(saved_model_dir):
        os.makedirs(saved_model_dir)
    torch.save(nn.state_dict(), saved_model_dir + "weights.pt")


def predict_bitmapping(nn, weights, x, resulution, threshold, denoise=False):
    nn.load_state_dict(torch.load(saved_model_dir + weights))
    nn.eval()
    generate_prediction(x, nn, resulution, threshold=threshold, denoise=denoise)


def mask(img_file, filter_size):
    img = imread(img_file)
    # edges = canny(img/255.)
    fill = ndi.binary_fill_holes(img)
    label_objects, nb_labels = ndi.label(fill)
    sizes = np.bincount(label_objects.ravel())
    mask_sizes = sizes > filter_size
    mask_sizes[0] = 0
    cleaned = mask_sizes[label_objects]
    return cleaned


def denoise(bitmapping, shape):
    y = np.genfromtxt(bitmapping)
    y = np.reshape(y, shape)
    y = y - 45
    y[0][0], y[0][shape[1]-1], y[shape[0]-1][shape[1]-1], y[shape[0]-1][0] = 0, 0, 0, 0
    for row in y:
        for i in range(len(row)):
            if row[i] < 0:
                row[i] = 0
            # if row[i] < 10:
            #     row[i] = row[i] / 2
            if row[i] >= 0:
                row[i] = row[i] * 2

    for i in range(1, shape[0]-1):
        for j in range(1, shape[1]-1):
            if ((y[i][j-1] == 0) & (y[i][j+1] == 0)) or ((y[i-1][j] == 0) & (y[i+1][j] == 0)):
                y[i][j] = 0

            else:
                y[i][j] = (y[i][j-1] + y[i][j+1]) / 2 + (y[i-1][j] + y[i+1][j])/10

    cv2.imwrite("ml/processed_Kevin.png", y)


if __name__ == "__main__":
    load_data(data_dir)
    dimension = len(data[0])

    nn = model.Model(dimension).double()

    # Initialize weights uniformly from -1 to 1
    nn.apply(weights_init_uniform)

    train_loader, val_loader = split_load_data(data, labels)

    # train_save(nn, train_loader, val_loader)
    predict_bitmapping(nn, weights='weights.pt', x=input, resulution=(50, 53),
                       threshold=False, denoise=False)

    denoise('ml/bitmapping.csv', (50, 53))

    # result = mask('ml/prediction_Kevin.png', mask_threshold)
    # io.imshow(result)
    # plt.show()


