import numpy as np
import torch, cv2
import albumentations as A
import matplotlib.pyplot as plt

from albumentations.pytorch import ToTensorV2

from configuration import device, classes

class Averager:

    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0

class SaveBestModel:

    def __init__(self, best_valid_loss=float('inf')):
        self.best_valid_loss = best_valid_loss

    def __call__(self, current_valid_loss, epoch, model, optimizer):

        if current_valid_loss < self.best_valid_loss:

            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")

            # Must specify absoloute path for saving training and validation results

            torch.save({'epoch': epoch+1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                        '/content/drive/MyDrive/fasterrccn_project/outputs/best_model.pth')

def collate_fn(batch):
    return tuple(zip(*batch))

# Train, Val and Test image transofrmation functions

def train_transform():
    return A.Compose([A.Flip(0.5),
                      A.RandomRotate90(0.5),
                      #A.MotionBlur(p=0.2),
                      A.MedianBlur(blur_limit=3, p=0.1),
                      A.Blur(blur_limit=3, p=0.1),
                      ToTensorV2(p=1.0)],
                      bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

def valid_transform():
    return A.Compose([ToTensorV2(p=1.0)],
                      bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

def test_transform():
    return A.Compose([ToTensorV2(p=1.0)],
                      bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

def show_transformed_image(train_loader):
    if len(train_loader) > 0:

        for i in range(1):

            images, targets = next(iter(train_loader))
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            boxes = targets[i]['boxes'].cpu().numpy().astype(np.int32)
            labels = targets[i]['labels'].cpu().numpy().astype(np.int32)
            sample = images[i].permute(1, 2, 0).cpu().numpy()

            for box_num, box in enumerate(boxes):
                cv2.rectangle(sample,
                              (box[0], box[1]),
                              (box[2], box[3]),
                              (0, 0, 255), 2)
                cv2.putText(sample, classes[labels[box_num]],
                            (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (0, 0, 255), 2)
            cv2.imshow(sample)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

def save_model(epoch, model, optimizer):
    # Must specify absoloute path for saving training and validation results

    torch.save({'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},
               '/content/drive/MyDrive/fasterrccn_project/outputs/last_model.pth')

def save_loss_plot(output_dir, train_loss, val_loss):

    figure_1, train_ax = plt.subplots()
    figure_2, valid_ax = plt.subplots()
    train_ax.plot(train_loss, color='tab:blue')
    train_ax.set_xlabel('iterations')
    train_ax.set_ylabel('train loss')
    valid_ax.plot(val_loss, color='tab:red')
    valid_ax.set_xlabel('iterations')
    valid_ax.set_ylabel('validation loss')
    figure_1.savefig(output_dir + '/train_loss.png')
    figure_2.savefig(output_dir + '/valid_loss.png')
    print('Saving completed')
    plt.close('all')
