import torch, time

from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from configuration import device, num_workers, num_classes, visualise_transformed_images, num_epochs, output_dir, \
    learning_rate, gamma
from dataset import create_train_dataset, create_train_loader, create_validation_dataset, create_validation_loader
from model import create_model
from utils import Averager, save_model, save_loss_plot, show_transformed_image, SaveBestModel


# Standardised training function

def train(train_data_loader, model):
    print('Training')
    global train_itr
    global train_loss_list

    model.train()

    prog_bar = tqdm(train_data_loader, total=len(train_data_loader))

    for i, data in enumerate(prog_bar):
        optimiser.zero_grad()

        images, targets = data
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        train_loss_list.append(loss_value)
        train_loss_hist.send(loss_value)
        losses.backward()
        optimiser.step()
        train_itr += 1

        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")

    return train_loss_list


# Standardised validation function

def validate(valid_data_loader, model):
    print('Validating')
    global val_itr
    global val_loss_list

    prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))

    for i, data in enumerate(prog_bar):
        images, targets = data
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        val_loss_list.append(loss_value)
        val_loss_hist.send(loss_value)
        val_itr += 1

        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")

    return val_loss_list


# Mean Average Precision logged per epoch

def calculate_mAP(data_loader, model):

    metric_test = MeanAveragePrecision()
    preds_single = []
    targets_single = []

    for batch_idx, (images, targets) in enumerate(data_loader, 1):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        targets_single.extend(targets)
        model.eval()

        with torch.no_grad():
            pred = model(images)
        preds_single.extend(pred)

    metric_test.update(preds_single, targets_single)
    test_map = metric_test.compute()
    print(f"Validation Mean Average Precision: {test_map['map']:.3f}")

    return test_map['map']


# Execute model training

torch.cuda.empty_cache()

val_loss_list = []

if __name__ == '__main__':

    train_dataset = create_train_dataset()
    valid_dataset = create_validation_dataset()

    train_loader = create_train_loader(train_dataset, num_workers)
    valid_loader = create_validation_loader(valid_dataset, num_workers)

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(valid_dataset)}\n")

    # Optimiser call, calibration and specify base parameters

    model = create_model(num_classes=num_classes)
    model = model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]

    optimiser = torch.optim.Adam(params,
                                 lr=learning_rate)

    scheduler = StepLR(optimiser, step_size=30, gamma=gamma)

    # Averager class initiated for training loss

    train_loss_hist = Averager()
    val_loss_hist = Averager()
    train_itr = 1
    val_itr = 1
    train_loss_list = []
    val_loss_list = []

    model_name = 'model'

    if visualise_transformed_images:
        show_transformed_image(train_loader)

    save_best_model = SaveBestModel()

    # Training and Validation loop per epoch

    for epoch in range(num_epochs):
        scheduler.step()

        print(f"\nEpoch {epoch + 1} of {num_epochs}")
        train_loss_hist.reset()
        val_loss_hist.reset()
        start = time.time()
        train_loss = train(train_loader, model)
        val_loss = validate(valid_loader, model)

        print(f"Epoch #{epoch + 1} train loss: {train_loss_hist.value:.3f}")
        print(f"Epoch #{epoch + 1} validation loss: {val_loss_hist.value:.3f}")

        mAP = calculate_mAP(valid_loader, model)

        end = time.time()
        print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch}")

        save_best_model(val_loss_hist.value, epoch, model, optimiser)
        save_model(epoch, model, optimiser)
        save_loss_plot(output_dir, train_loss, val_loss)

        time.sleep(5)
