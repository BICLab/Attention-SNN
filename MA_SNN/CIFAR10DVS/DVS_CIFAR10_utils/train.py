import torch
from tqdm import tqdm


def train(config):
    config.train_loss = 0
    config.train_correct = 0

    bar_train = tqdm(total=len(config.train_loader))
    for batch_idx, (input, labels) in enumerate(config.train_loader):

        config.model.zero_grad()
        config.optimizer.zero_grad()

        input = input.float().to(config.device)

        labels = labels[:, 1, :].float().to(config.device)
        outputs = config.model(input)
        
        # macs, params = profile(config.model, inputs=(input, )) 
        # print(macs)
        loss = config.criterion(outputs, labels)

        _, predicted = torch.max(outputs.data, 1)
        _, labelTest = torch.max(labels.data, 1)

        config.train_correct += (predicted == labelTest).sum().item()

        config.train_loss += loss.item()

        loss.backward()

        config.optimizer.step()
        bar_train.update()
        bar_train.set_description("Train:Epoch[%d/%d]" % (config.epoch + 1, config.num_epochs))
        bar_train.set_postfix(Loss=loss.item())

    bar_train.close()
