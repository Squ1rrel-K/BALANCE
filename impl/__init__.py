import gc
import torch
from opacus import PrivacyEngine
from torch.autograd import Variable
import util.easy_fed as ef
from util import count_loss
def train_client_DP_reg(client, model, optim, lr, loss_func, model_state, device, noise_multiplier=1.1,
                        max_grad_norm=1.0):
    sum_loss = 0
    size = 0
    local_model = model().to(device)
    local_model.load_state_dict(model_state)
    optimizer = optim(local_model.parameters(), lr=lr)
    privacy_engine = PrivacyEngine()
    local_model, optimizer, client = privacy_engine.make_private(
        module=local_model,
        optimizer=optimizer,
        data_loader=client,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )
    for data in client:
        inputs, labels = data
        inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)
        optimizer.zero_grad()
        outputs = local_model(inputs)
        loss = count_loss(loss_func, outputs, labels, device)
        loss.backward()
        optimizer.step()

        sum_loss += loss.data
        size += len(labels)
        del inputs
        del labels
        del loss
        gc.collect()
    client_update = ef.sub(model_state, ef.transfer(local_model.state_dict(), lambda key: key.replace('_module.', '')))
    del local_model
    gc.collect()
    return client_update, sum_loss, 0, size


def train_client_DP(client, model, optim, lr, loss_func, model_state, device, noise_multiplier=1.1, max_grad_norm=1.0):
    sum_loss = 0
    true_predict = 0
    size = 0
    local_model = model().to(device)
    local_model.load_state_dict(model_state)
    optimizer = optim(local_model.parameters(), lr=lr)
    privacy_engine = PrivacyEngine()
    local_model, optimizer, client = privacy_engine.make_private(
        module=local_model,
        optimizer=optimizer,
        data_loader=client,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )
    for data in client:
        inputs, labels = data
        inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)
        optimizer.zero_grad()
        outputs = local_model(inputs)
        loss = count_loss(loss_func, outputs, labels, device)
        loss.backward()
        optimizer.step()

        _, id = torch.max(outputs.data, 1)
        sum_loss += loss.data
        true_predict += torch.sum(id == labels.data)
        size += len(labels)
        del inputs
        del labels
        del loss
        gc.collect()
    client_update = ef.sub(model_state, ef.transfer(local_model.state_dict(), lambda key: key.replace('_module.', '')))
    del local_model
    gc.collect()
    return client_update, sum_loss, true_predict, size

def train_client_reg(client, model, optim, lr, loss_func, model_state, device):
    sum_loss = 0
    size = 0
    local_model = model().to(device)
    local_model.load_state_dict(model_state)
    optimizer = optim(local_model.parameters(), lr=lr)
    for data in client:
        inputs, labels = data
        inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)
        optimizer.zero_grad()
        outputs = local_model(inputs)
        loss = count_loss(loss_func, outputs, labels, device)
        loss.backward()
        optimizer.step()
        sum_loss += loss.data
        size += len(labels)
    client_update = ef.sub(model_state, local_model.state_dict())
    return client_update, sum_loss, 0, size


def train_client(client, model, optim, lr, loss_func, model_state, device):
    sum_loss = 0
    true_predict = 0
    size = 0
    local_model = model().to(device)
    local_model.load_state_dict(model_state)
    optimizer = optim(local_model.parameters(), lr=lr)
    for data in client:
        inputs, labels = data
        inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)
        optimizer.zero_grad()
        outputs = local_model(inputs)
        loss = count_loss(loss_func, outputs, labels, device)
        loss.backward()
        optimizer.step()

        _, id = torch.max(outputs.data, 1)
        sum_loss += loss.data
        true_predict += torch.sum(id == labels.data)
        size += len(labels)
    client_update = ef.sub(model_state, local_model.state_dict())
    return client_update, sum_loss, true_predict, size

def get_test_acc_reg(global_model, inputs, labels, loss_func, device):
    global_model.eval()
    outputs = global_model(inputs)
    loss = count_loss(loss_func, outputs, labels, device)
    size = len(labels)
    test_loss = loss.item()
    global_model.train()
    return 0, test_loss, size


def get_test_acc(global_model, inputs, labels, loss_func, device):
    global_model.eval()
    outputs = global_model(inputs)
    loss = count_loss(loss_func, outputs, labels, device)
    _, id = torch.max(outputs.data, 1)
    test_correct = torch.sum(id == labels.data).item()
    size = len(labels)
    test_loss = loss.item()
    global_model.train()
    return test_correct, test_loss, size