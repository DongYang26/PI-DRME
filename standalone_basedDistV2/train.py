import copy
from model import ServerNet, ClientNet
from utils import *
from dataset import DataModule, merge_eachClient_maps
import os
from losses import loss_custom
from metrics import *


seed_everything(42)


class EqualUserSampler(object):
    def __init__(self, n, num_users):
        self.i = 0
        self.selected = n
        self.num_users = num_users
        self.get_order()

    def get_order(self):
        self.users = np.arange(self.num_users)

    def get_useridx(self):
        selection = list()
        for _ in range(self.selected):
            selection.append(self.users[self.i])
            self.i += 1
            if self.i >= self.num_users:
                self.get_order()
                self.i = 0
        return selection


epochs = 120  # total epochs
local_epochs_s = 3  # local epochs of each user at an iteration
local_epochs_i = 10
saveLossInterval = 1  # intervals to save loss
batchSize = 2048  # batchsize for training and evaluation
num_users = 9  # ! total users
num_activate_users = 9
lr = 1e-3  # learning rate
cudaIdx = "cuda:0"  # GPU card index
_device = torch.device(cudaIdx if torch.cuda.is_available() else "cpu")
num_workers = 16  # workers for dataloader
evaluation = True  # evaluation only if True
client_weights = [1 / num_activate_users for i in range(num_activate_users)]

c = 100
n_segments = 100
TV_WEIGHT = 1e-7
setup = 1
simulation = "DPM"
carsSimulation = "no"
secarsInput = "no"
cityMap = "complete"
submap_size = 64  # ! 64*64
criterion_s = loss_custom('share', batchSize, c,
                          n_segments, TV_WEIGHT, _device)
criterion_i = loss_custom('individual', batchSize, c,
                          n_segments, TV_WEIGHT, _device)

script_directory = os.path.dirname(os.path.abspath(__file__))
savedModels = os.path.join(script_directory, 'savedModels/')
setup_name = ['uniform', 'twoside', 'nonuniform'][setup-1]
model_directory = f"{savedModels}{setup_name}_{get_next_exp_index(setup_name, savedModels)}"
saved_model_directory = f"{savedModels}{setup_name}_{get_last_exp_index(setup_name, savedModels)}"


class FedAvgServer:  # used as a center
    def __init__(self, global_parameters, client_model, client_weights, _device):
        self.global_parameters = global_parameters
        self._device = _device
        self.client_weights = client_weights

        self.prev_models = []
        self.local_parameters = [
            copy.deepcopy(client_model) for i in range(len(client_weights))]

    def upload(self, local_parameters):
        self.local_parameters = copy.deepcopy(local_parameters)

    def download(self, index):
        clientModel = copy.deepcopy(self.local_parameters[index])
        return clientModel

    def get_prev_models(self):
        return self.prev_models


class Client:  # as a user
    def __init__(self, data_loader, user_idx):
        self.data_loader = data_loader
        self.user_idx = user_idx
        self.best_loss = float('inf')
        self.best_model = None

    def update_best_model(self, model, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_model = copy.deepcopy(model.state_dict())

    def save_best_model(self):
        if self.best_model is not None:
            save_model(self.best_model, model_directory, self.user_idx)
            print(f"Saved best model for client {self.user_idx}")

    def train(self, model, learningRate, idx, _device):  # training locally
        optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
        for local_epoch in range(1, local_epochs_s + 1):
            for i, (inps, gts) in enumerate(self.data_loader):
                inps = inps.float().to(_device)
                gts = gts.float().to(_device)
                optimizer.zero_grad()
                estiamtion, a = model(inps[:, 1:3, :, :], 'shareU')

                loss = criterion_s.get_loss(estiamtion, gts, inps)
                loss.backward()
                optimizer.step()
                # self.update_best_model(model, loss.item())
                print(
                    f"Client: {idx}({self.user_idx:2d}) Local Epoch S: [{local_epoch}][{i}/{len(self.data_loader)}]---- loss {loss.item():.4f}")
        for local_epoch in range(1, local_epochs_i + 1):
            for i, (inps, gts) in enumerate(self.data_loader):
                inps = inps.float().to(_device)
                gts = gts.float().to(_device)
                optimizer.zero_grad()
                a, estiamtion = model(inps[:, 1:3, :, :], 'individualU')
                loss = criterion_i.get_loss(estiamtion, gts, inps)
                loss.backward()
                optimizer.step()
                self.update_best_model(model, loss.item())
                print(
                    f"Client: {idx}({self.user_idx:2d}) Local Epoch I: [{local_epoch}][{i}/{len(self.data_loader)}]---- loss {loss.item():.4f}")
        self.save_best_model()


def activateClient(train_dataloaders, user_idxs):
    clients = []
    for i in range(len(user_idxs)):
        clients.append(Client(train_dataloaders[user_idxs[i]], user_idxs[i]))
    return clients


def train(train_dataloaders, user_idxs, server, learningRate, _device):
    clients = activateClient(
        train_dataloaders, user_idxs)
    client_params = []
    for i in range(len(clients)):
        clientModel = server.download(i).to(_device)  # !
        clientModel.train()
        clients[i].train(clientModel, learningRate, i, _device)
        client_params.append(clientModel.to(_device))
    server.upload(client_params)


def test(base_model, data_loaders, models, user_idx, _device):
    estiamtions = []  # the estimation of all users with the test
    ground_truths = []  # the ground truth of all users with the test

    with torch.no_grad():
        getloss, rmse, mae, var, mape = Recoder(), Recoder(), Recoder(), Recoder(), Recoder()
        for i,  (model_name, model_state_dict) in enumerate(models.items()):  # for each model
            print(f"Testdation index: {i}, Model name: {model_name}")
            model_state_dict = {k: v.to(_device)
                                for k, v in model_state_dict.items()}
            base_model.load_state_dict(model_state_dict)
            base_model.eval()
            data_loader = data_loaders[user_idx[i]]

            inps, gts = next(iter(data_loader))
            inps = inps.float().to(_device)
            gts = gts.float().to(_device)
            a, estiamtion = base_model(inps[:, 1:3, :, :], 'individualU')
            estiamtions.append(estiamtion)
            ground_truths.append(gts)
            loss = criterion_i.getMseLoss(estiamtion, gts)
            getloss.update(loss)
            rmse.update(RMSE(estiamtion, gts))
            mae.update(MAE(estiamtion, gts))
            var.update(explained_variance(estiamtion, gts))
            mape.update(MAPE(estiamtion, gts))

        merged_estiamtion = merge_eachClient_maps(estiamtions, 256, 49)
        merged_gst = merge_eachClient_maps(ground_truths, 256, 49)

    print(f"Test_loss: {getloss.avg(): .4f}-RMSE: {(rmse.avg()): .4f}-MAE: {(mae.avg()): .4f}-ExplainedVar: {(var.avg()): .4f}-MAPE: {(mape.avg()): .4f}", flush=True)

    save_map_eachClient(f'{script_directory}/testVisualization/', setup_name,
                        merged_estiamtion, merged_gst, batchSize)


def train_main(server_model, client_model, client_weights, lr, sampler, fix, low, high, _device):
    if not os.path.exists(f'{model_directory}'):
        os.makedirs(f'{model_directory}')

    train_dataloaders = DataModule(
        num_activate_users, batchSize, simulation, carsSimulation, secarsInput, cityMap, num_workers, fix, low, high, mode='train')

    server = FedAvgServer(server_model, client_model, client_weights, _device)

    for epoch in range(1, epochs + 1):  # start training
        print('--------- Epoch {:<3d}-----------'.format(epoch))
        user_idx = sampler.get_useridx()
        if epoch % 50 == 0 and epoch != 0:
            lr *= 0.1

        train(train_dataloaders, user_idx, server, lr, _device)

        # models = load_pth_files(model_directory)
        # valid(valid_dataloaders, models, user_idx, epoch)  #  codes like test method


def test_main(clients_model, sampler, fix, low, high, _device):
    test_dataloaders = DataModule(
        num_activate_users, batchSize, simulation, carsSimulation, secarsInput, cityMap, num_workers, fix, low, high, mode='test')

    user_idx = sampler.get_useridx()
    models = load_pth_files(saved_model_directory)
    test(clients_model, test_dataloaders, models, user_idx, _device)


if __name__ == '__main__':
    print(
        f"setup:{setup};num_users:{num_users};num_activate_users:{num_activate_users};epochs:{epochs}")
    if setup == 1:
        fix, low, high = 655, 10, 300
    elif setup == 2:
        fix, low, high = 1, 655, 655*5
    else:
        fix, low, high = 0, 655, 655*5
    sampler = EqualUserSampler(num_activate_users, num_users)

    serverModel = ServerNet().to(_device)
    clientModel = ClientNet().to(_device)
    train_main(serverModel, clientModel, client_weights, lr,
               sampler, fix, low, high, _device)

    # _device = torch.device("cpu")  # If CUDA memory is sufficient, use cuda
    # clientModel = ClientNet().to(_device)
    # test_main(clientModel, sampler, fix, low, high, _device)

    # send_message()
