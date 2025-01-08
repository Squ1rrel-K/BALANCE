import copy

import torch

import impl.setting_impl as setting
import meta_define.setting
import util.parser
from impl.fl_framework_client_train_impl import ClientNormal, ClientAdv
from impl.fl_framework_impl import FedAvg
from impl.fl_task_impl import Classification
from meta_define.attack import Strategic, Attack
from meta_define.fl import ClientTrain, Framework, TaskType
from util import get_classes, auto_match, split_dataset_to_client
from util.log import Report


def train(model, client_data, test_dataset, epochs, lr,
          client_train: ClientTrain, framework: Framework, task_type: TaskType, optim,
          loss_func=torch.nn.CrossEntropyLoss(), device=torch.device('cuda'), report=Report()):
    if task_type is None:
        task_type = Classification()
    if client_train is None:
        client_train = ClientNormal()
    if framework is None:
        framework = FedAvg()
    framework.loss_func = loss_func
    framework.test_dataset = test_dataset
    framework.model = model
    framework.device = device
    framework.load()
    client_train.set_task_type(task_type)
    report.log["eval_records"] = []
    for epoch in range(epochs):
        updates = framework.train_clients(client_train, context={"lr": lr, "model": model, "client_data": client_data,
                                                                 "epoch": epoch, "epochs": epochs, "optim": optim,
                                                                 "loss_func": loss_func, "device": device})
        update = framework.aggregate({'updates': updates})
        framework.update({'update': update})
        evaluation = framework.evaluate({"task": task_type})
        report.log["eval_records"].append(evaluation)
        print(evaluation)
    report.global_record(
        context={"client_size": len(client_data), "epochs": epochs, "criterion": loss_func.__class__.__name__,
                 "task": client_train.__class__.__name__, "framework": framework.__class__.__name__})
    return framework.get_global_model()


def train_with_attack(model, client_data, test_dataset, epochs, lr, strategic: Strategic, attack: Attack, adv_rate,
                      client_train: ClientTrain, framework: Framework, task_type: TaskType, optim,
                      loss_func=torch.nn.CrossEntropyLoss(), device=torch.device('cuda'), report=Report()):
    if task_type is None:
        task_type = Classification()
    if client_train is None:
        client_train = ClientAdv()
    if framework is None:
        framework = FedAvg()
    framework.loss_func = loss_func
    framework.test_dataset = test_dataset
    framework.model = model
    framework.device = device
    framework.load()
    client_train.set_task_type(task_type)
    n = len(client_data)
    adv_n = int(n * adv_rate)
    report.log["eval_records"] = []
    for epoch in range(epochs):
        tag = []
        context = {"lr": lr, "model": model, "client_data": client_data,
                   "adv_n": adv_n, "tag": tag, 'global_model': framework.get_global_model(),
                   "epoch": epoch, "epochs": epochs, "attack": attack,
                   "loss_func": loss_func, "device": device}
        attack.assign(context)
        updates = framework.train_clients(client_train, context={"lr": lr, "model": model, "client_data": client_data,
                                                                 "adv_n": adv_n, "tag": tag, "optim": optim,
                                                                 "epoch": epoch, "epochs": epochs, "attack": attack,
                                                                 "loss_func": loss_func, "device": device})
        filtered_updates = strategic.aggregate({'global_model': framework.get_global_model(),
                                                'updates': updates})
        update = framework.aggregate({'updates': filtered_updates})
        framework.update({'update': update})
        report.byzantine_record(context={"updates": updates, "tag": tag, "strategic": strategic})
        evaluation = framework.evaluate({"task": task_type})
        report.log["eval_records"].append(evaluation)
        print(evaluation)
    report.global_record(context={"client_size": n, "epochs": epochs, "criterion": loss_func.__class__.__name__,
                                  "task": client_train.__class__.__name__, "framework": framework.__class__.__name__,
                                  "strategic": strategic.__class__.__name__, "adversary_rate": adv_rate})
    return framework.get_global_model()


def start(args: dict):
    settings = get_classes(setting, meta_define.setting.Setting)
    settings.sort(key=lambda o: o.order)
    parse_args = copy.deepcopy(args)
    for s in settings:
        if s is not meta_define.setting.Setting:
            auto_match(s(), args, parse_args)
    framework = parse_args["framework"]
    task = parse_args["fl_settings"]["task"]
    n = parse_args["fl_settings"]["client_size"]
    lr = parse_args["fl_settings"]["lr"]
    epochs = parse_args["fl_settings"]["epochs"]
    train_dataset, test_dataset = parse_args["fl_settings"]["dataset"]
    model = parse_args["fl_settings"]["model"]
    clients = split_dataset_to_client(n, train_dataset)
    loss_func = parse_args["fl_settings"]["loss_func"]
    optim = parse_args["fl_settings"]["optim"]
    device = parse_args["fl_settings"]["device"]
    report = Report()
    train_func = lambda: train(model, clients, test_dataset, epochs, lr, ClientNormal(),
                               framework, task, optim=optim,
                               loss_func=loss_func, device=device, report=report)
    if "byzantine_settings" in args.keys():
        context = {}
        context.update(args)
        context["client_data_size"] = int(len(train_dataset) / n)
        context["test_dataset"] = test_dataset
        context["model"] = model
        attack = parse_args["byzantine_settings"]["byzantine_attack"]
        byzantine_defend = parse_args["byzantine_settings"]["byzantine_defend"]
        train_func = lambda: train_with_attack(model, clients, test_dataset, epochs, lr,
                                               byzantine_defend,
                                               attack,
                                               args["byzantine_settings"]["adv_rate"],
                                               ClientAdv(), framework, task, optim=optim,
                                               loss_func=loss_func, device=device, report=report)

    train_func()
    report.log["meta"] = args
    print(report.log)
    report.save(args["other_settings"]["report_dir"], args["other_settings"]["report_name"]+".npy")


if __name__ == '__main__':
    args = util.parser.read_json("../config/test.json")
    start(args)
