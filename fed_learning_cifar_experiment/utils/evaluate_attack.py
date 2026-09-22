import torch

from fed_learning_cifar_experiment.utils.backdoor_attack import add_trigger
from fed_learning_cifar_experiment.task import set_weights, test_eval
from fed_learning_cifar_experiment.utils.logger import append_centralized_round


def evaluate_asr(model, test_data, target_label, device=None, num_samples=1000):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    correct, total = 0, 0

    with torch.no_grad():
        for batch in test_data:
            if isinstance(batch, dict):
                imgs, labels = batch["img"], batch["label"]
            else:
                imgs, labels = batch

            imgs, labels = imgs.to(device), labels.to(device)

            mask = labels != target_label
            if mask.sum() == 0:
                continue

            imgs = torch.stack([add_trigger(img) for img in imgs[mask]])

            outputs = model(imgs)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == target_label).sum().item()
            total += mask.sum().item()

            if total >= num_samples:
                break

    return correct / total if total > 0 else 0.0

def get_evaluate_fn(model, test_data, target_label=2):
    def evaluate(server_round, parameters, config):
        set_weights(model, parameters)
        num_clients = config.get("num-clients")
        simulation_id = config.get("simulation-id")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        loss, acc = test_eval(model, test_data, device=device)

        asr = evaluate_asr(model, test_data, target_label, device=device)
        append_centralized_round(simulation_id, server_round, loss, acc, asr, num_clients)
        return loss, {"mta": acc, "asr": asr}

    return evaluate
