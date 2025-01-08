import torch

from meta_define.fl import ClientTrain
from util import cover


class ClientAdv(ClientTrain):

    def train(self, context: dict):
        ci = context["i"]
        adv_n = context["adv_n"]
        tag = context["tag"]
        epoch = context["epoch"]
        epochs = context["epochs"]
        attack = context["attack"]
        context = cover(context)
        if ci >= adv_n:
            tag.append(0)
            client_update, sum_loss, true_predict, size = self.task_type.train(context)
            print('[%d,%d] loss:%.03f correct:%.03f%%' % (
                epoch + 1, epochs, sum_loss / size, 100 * true_predict / size))
        else:
            tag.append(1)
            client_update = attack.gen_gradient(context)
            print('[%d,%d] %s' % (epoch + 1, epochs, "malicious client..."))
        return client_update


class ClientNormal(ClientTrain):

    def train(self, context: dict):
        epoch = context["epoch"]
        epochs = context["epochs"]
        context = cover(context)
        context.update({'optim': torch.optim.Adam})
        client_update, sum_loss, true_predict, size = self.task_type.train(context)
        print('[%d,%d] loss:%.03f correct:%.03f%%' % (
            epoch + 1, epochs, sum_loss / size, 100 * true_predict / size))
        return client_update