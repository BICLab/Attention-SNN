import torch
from tqdm import tqdm


def test(config):
    config.test_loss = 0
    config.test_correct = 0
    bar_test = tqdm(total=len(config.test_loader))
    for batch_idx, (input, labels) in enumerate(config.test_loader):

        b = input.size()[0]

        input = input.reshape(
            b * config.clip,
            input.size()[2],
            input.size()[3],
            input.size()[4],
            input.size()[5])
        input = input.float().to(config.device)

        labels = labels.reshape(
            b * config.clip,
            labels.size()[2],
            labels.size()[3])
        labels = labels[:, 1, :].float().to(config.device)
        outputs = config.model(input)

        loss = config.criterion(outputs, labels)

        _, predicted = torch.max(outputs.data, 1)
        _, labelTest = torch.max(labels.data, 1)

        for i in range(b):
            predicted_clips = predicted[i * config.clip:(i + 1) * config.clip]
            labelTest_clips = labelTest[i * config.clip:(i + 1) * config.clip]
            test_clip_correct = (
                    predicted_clips == labelTest_clips).sum().item()
            if test_clip_correct / config.clip > 0.5:
                config.test_correct += 1

        config.test_loss += loss.item() / config.clip
        bar_test.update()
        bar_test.set_description("Test:Epoch[%d/%d]" % (config.epoch + 1, config.num_epochs))
        bar_test.set_postfix(Loss=loss.item())

    bar_test.close()
