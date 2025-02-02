import os
import torch
import random
import numpy as np
import torch.multiprocessing
import matplotlib.pyplot as plt
import requests


torch.multiprocessing.set_sharing_strategy('file_system')


def backward(optimizer, loss):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def db_to_natural(x_variable_db):
    """
        Arguments:
              x_variable_db: must be an np array
        """
    return 10 ** (x_variable_db / 10)


def dbm_to_db(x_variable_dbm):
    """
        Arguments:
            x_variable_dbm: must be an np array
      """
    return x_variable_dbm - 30


def dbm_to_natural(x_variable_dbm):
    return db_to_natural(dbm_to_db(x_variable_dbm))


def natural_to_db(x_variable_nat):
    """
      Arguments:
            x_variable_nat: must be an np array
      """
    return 10 * np.log10(x_variable_nat)


def db_to_natural(x_variable_db):
    """
        Arguments:
              x_variable_db: must be an np array
        """
    return 10 ** (x_variable_db / 10)


def save_array_as_pdf(array, filename):
    """
        Save a numpy array as PDF with coordinate information.

    Parameters:
        array (numpy.ndarray): The numpy array to be saved.
        filename (str): The filename for the saved PDF.
    """
    fig, ax = plt.subplots()

    if len(array.shape) == 2:
        im = ax.imshow(array)
    elif len(array.shape) == 3:
        im = ax.imshow(array[:, :, 0])

    plt.xlabel('X')
    plt.ylabel('Y')
    cbar = plt.colorbar(im)
    plt.savefig(filename)
    plt.close()


def merge_and_save_arrays_as_pdf(array_3d, array_2d, filename):
    """
    Merge two numpy arrays and save as PDF with coordinate information.

    Parameters:
        array_3d (numpy.ndarray): The 3D numpy array.
        array_2d (numpy.ndarray): The 2D numpy array.
        filename (str): The filename for the saved PDF.
    """
    assert array_3d.shape[:2] == array_2d.shape

    fig, ax = plt.subplots()
    im = ax.imshow(array_3d[:, :, 0] + array_2d)

    plt.xlabel('X')
    plt.ylabel('Y')
    cbar = plt.colorbar(im)
    plt.savefig(filename)
    plt.close()


class Recoder(object):
    def __init__(self):
        self.last = 0
        self.values = []
        self.nums = []

    def update(self, val, n=1):
        self.last = val
        self.values.append(val)
        self.nums.append(n)

    def avg(self):
        sum = np.sum(np.asarray(self.values)*np.asarray(self.nums))
        count = np.sum(np.asarray(self.nums))
        return sum/count


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def toNP(tensor):
    return tensor.detach().cpu().numpy()


def makeDIRs(folder):
    if not os.path.exists(f'models/{folder}/'):
        os.makedirs(f'models/{folder}/')
    if not os.path.exists(f'results/{folder}/'):
        os.makedirs(f'results/{folder}/')


def checkPoint(epoch, epochs, model, loss, saveLossInterval):

    # if epoch % saveModelInterval == 0 or epoch == epochs:
    #     torch.save(model.state_dict(), f'models/model' +
    #                str(epoch) + '.pth', _use_new_zipfile_serialization=False)
    if epoch % saveLossInterval == 0 or epoch == epochs:
        np.save(f'results/val_loss.npy', np.asarray(loss))


def save_data(dim):
    sequence = np.arange(0, 2304)
    array = sequence.reshape((dim, dim))
    group = []

    for i in range(9):
        a = []
        group.append(a)

    for j in range(48):

        if int(j / 16) == 0:
            x = 0
        elif int(j / 16) == 1:
            x = 3
        elif int(j / 16) == 2:
            x = 6

        for i in range(48):

            if i < 16:
                group[x].append(array[j, i])
            elif i < 32:
                group[x + 1].append(array[j, i])
            else:
                group[x + 2].append(array[j, i])

    return group


# [(180,32,32,2) (180,1024,1) (180,1024)] [(20,32,32,2) (20,1024,1) (20,1024)]
def coordinate(all_data):
    train_x = all_data[0][0]
    num, X, Y, _ = train_x.shape
    array = np.zeros(len(train_x.shape))
    sequence = np.arange(0, X*Y)
    # num_map = np.zeros(len(sequence))
    num_map = []

    for i in range(num):

        for j in range(len(sequence)):
            x, y = divmod(j, X)
            array[0] = x
            array[1] = y
            array[2] = train_x[i][x][y][0]
            array[3] = train_x[i][x][y][1]
            num_map.append(array)
        my_array = np.array(num_map)
        num_map.clear()
        np.save('data/train/{}_user.npy'.format(i), my_array)

    print(my_array.shape)


def load_pth_files(directory):
    # Get all eligible file names
    pth_files = [f for f in os.listdir(directory) if f.startswith(
        'client_') and f.endswith('_best.pth')]
    # sort the file by its names
    pth_files.sort(key=lambda x: int(x.split('_')[1]))
    loaded_models = {}
    for file_name in pth_files:
        file_path = os.path.join(directory, file_name)
        model = torch.load(file_path)
        loaded_models[file_name] = model

    return loaded_models


def get_next_exp_index(setup_name, name):
    """
    This function generates the next available experiment index for a given setup name.
    """
    exp_index = 0
    while os.path.exists(f"{name}{setup_name}_{exp_index}"):
        exp_index += 1
    return exp_index


def get_last_exp_index(setup_name, name):
    """
    This function gets the last experiment index for a given setup name.
    """
    exp_index = 0
    while os.path.exists(f"{name}{setup_name}_{exp_index}"):
        exp_index += 1
    return exp_index-1


def save_map_eachClient(name, setup_name, test_step_ests, test_step_gts, batchsize):  # (9,4,256,256)
    exp_path = f"{name}{setup_name}_{get_next_exp_index(setup_name, name)}"
    os.makedirs(exp_path, exist_ok=True)

    test_step_ests = np.array(test_step_ests)
    test_step_gts = np.array(test_step_gts)
    num_maps = test_step_gts.shape[1]
    num_clients = test_step_ests.shape[0]
    for i in range(num_clients):
        for m in range(num_maps):
            # get alone sample
            output_image = test_step_ests[i, m]  # (256, 256)
            target_image = test_step_gts[i, m]   # (256, 256)

            #
            output_image = (output_image - output_image.min()) / \
                (output_image.max() - output_image.min())
            target_image = (target_image - target_image.min()) / \
                (target_image.max() - target_image.min())

            #
            output_path = os.path.join(exp_path, f'test_out_{i}_{m}.png')
            target_path = os.path.join(exp_path, f'test_outgt{i}_{m}.png')

            plt.imsave(output_path, output_image, cmap='viridis')
            plt.imsave(target_path, target_image, cmap='viridis')

        print(f"Saved images for batch {i}")

    print(f"All images saved in {exp_path}")


def save_map(name, setup_name, test_step_ests, test_step_gts, batchsize):
    exp_path = f"{name}{setup_name}_{get_next_exp_index(setup_name, name)}"
    os.makedirs(exp_path, exist_ok=True)

    test_step_ests = np.array(test_step_ests)
    test_step_gts = np.array(test_step_gts)

    for i in range(min(batchsize, test_step_ests.shape[0])):
        # get alone sample
        output_image = test_step_ests[i, 0]  # (256, 256)
        target_image = test_step_gts[i, 0]   # (256, 256)

        #
        output_image = (output_image - output_image.min()) / \
            (output_image.max() - output_image.min())
        target_image = (target_image - target_image.min()) / \
            (target_image.max() - target_image.min())

        #
        output_path = os.path.join(exp_path, f'test_out_{i}.png')
        target_path = os.path.join(exp_path, f'test_outgt_{i}.png')

        plt.imsave(output_path, output_image, cmap='viridis')
        plt.imsave(target_path, target_image, cmap='viridis')

        print(f"Saved images for batch {i}")

    print(f"All images saved in {exp_path}")


def save_model(best_model, model_directory, user_idx):
    torch.save(best_model, model_directory+'/' + f'client_{user_idx}_best.pth')


