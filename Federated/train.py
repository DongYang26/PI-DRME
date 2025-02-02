import copy
from model import RadioNet
from utils import *
from dataset import DataModule, FederatedRadioUNet, merge_with_overlap, merge_eachClient_maps
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


epochs = 150  # total epochs
local_epochs = 3  # local epochs of each user at an iteration
saveLossInterval = 1  # intervals to save loss
batchSize = 2048  # batchsize for training and evaluation
num_users = 9  # ! total users
num_activate_users = 9
lr = 3e-3  # learning rate
cudaIdx = "cuda:0"  # GPU card index
_device = torch.device(cudaIdx if torch.cuda.is_available() else "cpu")
num_workers = 16  # workers for dataloader
evaluation = True  # evaluation only if True


lossPhase = 'first'  # ! 'first' or'second'
c = 100
n_segments = 100
TV_WEIGHT = 1e-7
setup = 1
simulation = "DPM"
carsSimulation = "no"
secarsInput = "no"
cityMap = "complete"
submap_size = 64  # ! 128*128
criterion = loss_custom(lossPhase, batchSize, c,
                        n_segments, TV_WEIGHT, _device)

script_directory = os.path.dirname(os.path.abspath(__file__))
savedModels = os.path.join(script_directory, 'savedModels/')
setup_name = ['uniform', 'twoside', 'nonuniform'][setup-1]
model_directory = f"{savedModels}{setup_name}_{get_next_exp_index(setup_name, savedModels)}"
saved_model_directory = f"{savedModels}{setup_name}_{get_last_exp_index(setup_name, savedModels)}"


class FedAvgServer:  # used as a center
    def __init__(self, global_parameters, _device):
        self.global_parameters = global_parameters
        self._device = _device

    def download(self, user_idx):
        local_parameters = []
        for i in range(len(user_idx)):
            local_parameters.append(copy.deepcopy(self.global_parameters))
        return local_parameters

    def upload(self, local_parameters):
        for i, (k, v) in enumerate(self.global_parameters.items()):
            tmp_v = torch.zeros_like(v).to(self._device)
            for j in range(len(local_parameters)):
                tmp_v += local_parameters[j][k].to(self._device)
            tmp_v = tmp_v / len(local_parameters)  # FedAvg
            self.global_parameters[k] = tmp_v


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
        for local_epoch in range(1, local_epochs + 1):
            for i, (inps, gts) in enumerate(self.data_loader):
                inps = inps.float().to(_device)
                gts = gts.float().to(_device)
                optimizer.zero_grad()
                estiamtion = model(inps[:, 1:3, :, :])

                loss = criterion.get_loss(estiamtion, gts, inps)
                loss.backward()
                optimizer.step()
                self.update_best_model(model, loss.item())
                print(
                    f"Client: {idx}({self.user_idx:2d}) Local Epoch: [{local_epoch}][{i}/{len(self.data_loader)}]---- loss {loss.item():.4f}")
        self.save_best_model()


def activateClient(train_dataloaders, user_idx, server):
    local_parameters = server.download(user_idx)
    clients = []
    for i in range(len(user_idx)):
        clients.append(Client(train_dataloaders[user_idx[i]], user_idx[i]))
    return clients, local_parameters


def train(train_dataloaders, user_idx, server, global_model, learningRate, _device):
    clients, local_parameters = activateClient(
        train_dataloaders, user_idx, server)
    for i in range(len(user_idx)):
        model = RadioNet().to(_device)  # !
        model.load_state_dict(local_parameters[i])
        model.train()
        clients[i].train(model, learningRate, i, _device)
        local_parameters[i] = model.to(_device).state_dict()
    server.upload(local_parameters)
    global_model.load_state_dict(server.global_parameters)


def valid(data_loader, models, user_idx, epoch):
    estiamtions = []  # the estimation of all users with the validation
    ground_truths = []  # the ground truth of all users with the validation
    merged_estiamtions = []
    merged_gsts = []
    with torch.no_grad():
        getloss, rmse, mae, var, mape = Recoder(), Recoder(), Recoder(), Recoder(), Recoder()
        for i,  (model_name, model) in enumerate(models.items()):
            print(f"Testdation index: {i}, Model name: {model_name}")
            model.eval()
            data_loader = data_loader[user_idx[i]]

            for i, (inps, gts) in enumerate(data_loader):
                if i > 0:
                    raise ValueError(
                        "Data loader contains more than one batch.")
                inps = inps.float().to(_device)
                gts = gts.float().to(_device)
                estiamtion = model(inps)
                estiamtions.append(estiamtion)
                ground_truths.append(gts)
                loss = criterion.getMseLoss(estiamtion, gts, inps)
                getloss.update(loss)
                rmse.update(RMSE(estiamtion, gts))
                mae.update(MAE(estiamtion, gts))
                var.update(explained_variance(estiamtion, gts))
                mape.update(MAPE(estiamtion, gts))

            merged_estiamtion = merge_with_overlap(estiamtions, 256, num_users)
            merged_gst = merge_with_overlap(ground_truths, 256, num_users)
            merged_estiamtions.append(merged_estiamtion)
            merged_gsts.append(merged_gst)

    print(f"Global Epoch: {epoch}-Val_loss: {getloss.avg(): .4f}-RMSE: {(rmse.avg()): .4f}-MAE: {(mae.avg()): .4f}-ExplainedVar: {(var.avg()): .4f}-MAPE: {(mape.avg()): .4f}", flush=True)
    save_map(f'{script_directory}/validVisualization/', setup_name,
             merged_estiamtions, merged_gsts, batchSize)


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
            estiamtion = base_model(inps[:, 1:3, :, :])
            estiamtions.append(estiamtion)
            ground_truths.append(gts)
            loss = criterion.getMseLoss(estiamtion, gts)
            getloss.update(loss)
            rmse.update(RMSE(estiamtion, gts))
            mae.update(MAE(estiamtion, gts))
            var.update(explained_variance(estiamtion, gts))
            mape.update(MAPE(estiamtion, gts))

        merged_estiamtion = merge_eachClient_maps(estiamtions, 256, 49)
        merged_gst = merge_eachClient_maps(ground_truths, 256, 49)
    print(f"Test_loss: {getloss.avg(): .4f}-RMSE: {(rmse.avg()): .4f}-MAE: {(mae.avg()): .4f}-ExplainedVar: {(var.avg()): .4f}-MAPE: {(mape.avg()): .4f}", flush=True)

    save_map_eachClient(f'{script_directory}/testVisualization/',
                        setup_name, merged_estiamtion, merged_gst, batchSize)


def train_main(model, sampler, fix, low, high, _device):
    if not os.path.exists(f'{model_directory}'):
        os.makedirs(f'{model_directory}')
    train_dataloaders = []

    train_dataloaders = DataModule(
        num_activate_users, batchSize, simulation, carsSimulation, secarsInput, cityMap, num_workers, fix, low, high, mode='train')

    global_parameters = model.state_dict()

    server = FedAvgServer(global_parameters, _device)

    for epoch in range(1, epochs + 1):  # start training
        user_idx = sampler.get_useridx()
        if lr < 50:
            train(train_dataloaders, user_idx, server, model, lr, _device)
        elif lr < 150:
            train(train_dataloaders, user_idx,
                  server, model, lr * 0.1, _device)
        else:
            train(train_dataloaders, user_idx,
                  server, model, lr * 0.02, _device)

        # models = load_pth_files(model_directory)
        # valid(valid_dataloaders, models, user_idx, epoch)  #  codes like test method


def test_main(model, sampler, fix, low, high, _device):
    test_dataloaders = DataModule(
        num_activate_users, batchSize, simulation, carsSimulation, secarsInput, cityMap, num_workers, fix, low, high, mode='test')

    user_idx = sampler.get_useridx()
    models = load_pth_files(saved_model_directory)
    test(model, test_dataloaders, models, user_idx, _device)


if __name__ == '__main__':
    print(f"setup:{setup};num_users:{num_users};num_activate_users:{num_activate_users};epochs:{epochs};local_epochs:{local_epochs}")

    if setup == 1:
        fix, low, high = 655, 10, 300
    elif setup == 2:
        fix, low, high = 1, 655, 655*10
    else:
        fix, low, high = 0, 655, 655*10
    sampler = EqualUserSampler(num_activate_users, num_users)

    model = RadioNet().to(_device)
    train_main(model, sampler, fix, low, high, _device)

    # _device = torch.device("cpu")  # If CUDA memory is sufficient, use cuda
    # model = RadioNet().to(_device)
    # test_main(model, sampler, fix, low, high, _device)
    
