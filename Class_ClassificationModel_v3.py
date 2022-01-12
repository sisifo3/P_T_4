import os
import torch
import pandas as pd
import torchvision
from torchvision import datasets, models, transforms

def ClassificationModelResNet():
    data_transforms = {
        'test': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    def test(test_loader, model, checkpoint, device):
        if checkpoint is not None:
            model.load_state_dict(torch.load(checkpoint, map_location='cpu'))

        gt = torch.FloatTensor().to(device)
        pred = torch.FloatTensor().to(device)

        model.eval()
        with torch.no_grad():
            for i, (data, target) in enumerate(test_loader['test']):
                #out = torchvision.utils.make_grid(data)
                target = target.to(device)
                print("target", target)
                gt = torch.cat((gt, target.float()), 0).to(device)
                out = model(data)
                out = torch.sigmoid(out)
                pred = torch.cat((pred, out), 0)

        return pred, gt

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #data_dir = '/home/sisifo/PycharmProjects/ML/venv/dataset_C/'
    data_dir = "/home/sisifo/PycharmProjects/ML/venv/Utils_3/dataset_C"

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                  shuffle=False, num_workers=2) for x in ['test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['test']}

    class_names = image_datasets['test'].classes
    print(class_names)

    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2)
    model = model.to(device)

    checkpoint = '/home/sisifo/Desktop/ResNet18.pth'

    pred, gt = test(dataloaders, model, checkpoint, device)

    # print("pred", pred)
    #print("gt", gt)

    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0

    pred_ten_f = []
    for val in pred: pred_ten_f.append(val[1])

    #print("pred", pred_ten_f)

    ####count########
    # ['sitting_people', 'standing_people']

    # sitting_people = 0
    # standing_people = 1
    c_sit_peo = 0
    c_stt_peo = 0
    for i, (val) in enumerate(pred_ten_f):
        if val == 0:
            c_sit_peo = c_sit_peo + 1
        else:
            c_stt_peo = c_stt_peo + 1
    #print("sitting_people: ", c_sit_peo)
    #print("c_stt_peo :", c_stt_peo)
    return c_sit_peo, c_stt_peo

