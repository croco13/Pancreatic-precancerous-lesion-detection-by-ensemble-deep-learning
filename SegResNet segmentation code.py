# Import necessary libraries
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
import segmentation_models_pytorch as smp
from torchvision.transforms import ToTensor, ToPILImage, Resize, CenterCrop, ConvertImageDtype, Normalize
from torchmetrics import JaccardIndex
import torchmetrics.functional as tmf
import segmentation_models_pytorch as smp
import torchvision.models as models
import matplotlib.pyplot as plt
import importlib as ipl
import numpy as np
import random
import pickle
import os
import glob
import time
import urllib
from timeit import default_timer as timer
import gc
from zipfile import ZipFile
from PIL import Image, ImageColor



# check the PyTorch version;
print("PyTorch version: ", torch.__version__)
print("torchvision version: ", torchvision.__version__)

# check the GPU support; shold be yes
print("Is GPU available?: ", torch.cuda.is_available())


dataset_choice = "wsi"

# WSI
wsi_class = ["IPMN", "NOT_IPMN"]
wsi_color = [(255,255,255), (0,0,0)]

# WSI (If you choose 3-class segmentation)
#wsi_class = ["IPMN", "NOT_IPMN", "Other"]
#wsi_color = [(255,255,255), (128,128,128), (0,0,0)]



# DATA ROOT
filepaths = {
    'wsi': 'D:/Internship_wemmert/WSI_9',

}

# COMBINE
color_maps = {
    "wsi": wsi_color,


}

class_maps = {
    "wsi": wsi_class,

}

image_dirs = {
    "wsi": "images",

}

mask_dirs = {
    "wsi": "masks",

}

image_exts = {
    "wsi": ".png",

}


mask_exts = {
    "wsi": ".png",

}

filepath = filepaths[dataset_choice]
color_mapping = color_maps[dataset_choice]
classes = class_maps[dataset_choice]
image_dir = image_dirs[dataset_choice]
mask_dir = mask_dirs[dataset_choice]
image_ext = image_exts[dataset_choice]
mask_ext = mask_exts[dataset_choice]


# ******** IMPORTANT HELPER FUNCTIONS FOR THIS LAB ****************

# Show the contents of your dataset folder
# It is important to see how your dataset is organized.
# You can also do this using file explorer
def show_folder_structure(startpath):
    assert os.path.exists(startpath), "File path does not exist!"
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for j,f in enumerate(sorted(files[:3])):
            print('{}{}'.format(subindent, f))
            if j==2:
              print(subindent,'...')
              print('{}{}'.format(subindent, files[-1]))


# Display images and labels
def display_image(images, titles):
    f, axes = plt.subplots(1, len(images), sharey=True)
    for i in range(len(images)):
        axes[i].imshow(images[i])
        axes[i].set_title(titles[i], fontsize=8, color= 'blue')
    plt.show()


# Convert segmentation mask from RGB to Semantic channel.
# RGB channel = 3 (reg, green, blue)
# Semantic channel = N, where N = number of classes, one channel per class
def rgb_to_semantic(image, color_mapping):
    image_array = np.array(image)
    repeated_image = np.repeat(image_array[:, :, np.newaxis, :], len(color_mapping), axis=2) # [rgb channels] x number of classes
    repeated_mapping = np.repeat(np.array(list(color_mapping))[np.newaxis, np.newaxis, :, :], image_array.shape[0], axis=0) # [semantic channels] x number of classes
    maskND = np.all(repeated_image == repeated_mapping, axis=-1).astype(np.uint8) # Equality broadcast
    return maskND


# Convert segmentation mask with semantic channel to a single channel
# Use NumPy broadcasting to assign the keys to the matching pixels
# Each pixel takes the class categorical value
def nD_to_1D(maskND):
    mask1D = np.argmax(maskND, axis=-1)
    return mask1D


# Convert semantic channel mask to rgb channel image
# Create an array of RGB values corresponding to keys in the mapping
# And Map the keys in the image to their corresponding RGB values
def semantic_to_rgb(mapped_image, color_mapping):
    color_array = np.array(color_mapping, dtype=np.uint8)
    rgb_image = color_array[mapped_image]
    return rgb_image



# folder structure exploration

show_folder_structure(filepath)


# check data size

train_size = len(os.listdir(os.path.join(filepath, "train", image_dir)))
val_size = len(os.listdir(os.path.join(filepath, "val", image_dir)))
test_size = len(os.listdir(os.path.join(filepath, "test", image_dir)))

print("Size | train: {}, val: {}, test:{}".format(train_size, val_size, test_size))

# show one image and label to see what they are like, in actual sense, you need to check many examples.

selected_img_file = '19AG01438-23_0735.png'
selected_msk_file = '19AG01438-23_0735.png'

#selected_img_file = os.listdir(os.path.join(filepath, "train", image_dir))[1000]
#selected_msk_file = os.listdir(os.path.join(filepath, "train", mask_dir))[1000]

img1_url = os.path.join(filepath, "train", image_dir, selected_img_file)
msk1_url = os.path.join(filepath, "train", mask_dir, selected_msk_file)
rgb_img1 = Image.open(img1_url).convert("RGB")
rgb_msk1 = Image.open(msk1_url).convert("RGB")
rgb_img1 = np.array(rgb_img1)
rgb_msk1 = np.array(rgb_msk1)


display_image(images=[rgb_img1, rgb_msk1], titles=['image', 'mask'])


# Check your data shape and distribution
# This is very important to understand your data

print("Image shape = ", rgb_img1.shape)
print("Mask shape = ", rgb_msk1.shape)

print("Image distribution: [Min = {}, Mean = {}, Max = {}] ".format(rgb_img1.min(), rgb_img1.mean(), rgb_img1.max()))
print("Mask distribution: [Min = {}, Mean = {}, Max = {}] ".format(rgb_msk1.min(), rgb_msk1.mean(), rgb_msk1.max()))


# Data Label Processing
# You are to process your label to have it in a format that your model can use.
# You have seen the shape and it has RGB channel but your mode will need a semantic channel
# Semantic label means N channel where N = number of classes

# 1. Convert RGB channel to semantic channel mask.

semantic_mask_ND = rgb_to_semantic(rgb_msk1, color_mapping)

# 2. Convert semantic mask to single channel mask
# We can now visualize the converted semantic channel mask,
# So we convert it to single channel with each pixel having the channel index with maximum value
semantic_mask_1D = nD_to_1D(semantic_mask_ND)

# 3. Recover rgb mask
# We can convert the single channel easily to the RGB channel to visual the mask.
# If you didn't get back your original RGB mask, it means your label processing code is not correct.
recovered_rgb_msk1 = semantic_to_rgb(semantic_mask_1D, color_mapping)

# 4. Visualize
display_image(images=[rgb_img1, rgb_msk1, semantic_mask_1D, recovered_rgb_msk1],
              titles=['Image', 'RGB mask', "Semantic mask", "Reversed RGB mask"])


# NB: We will only need the semantic channel mask for model training, the rest is for visualization purpose.

# We define a dataset class that delivers images and correponding ground truth segmentation masks

class MyDataset(torch.utils.data.Dataset):
    # Dataset class will inherit torch.utils.data.Dataset
    # There are 3 most important function to overider here
    # 1. `init` function: This prepare your dataset like a stack of data that are indexable
    # 2. `len` function: This return the total number of data you have
    # 3. `getitem` function: This return individual (image, target) on each call
    # You can write other functions that can help these 3 fulfill their duties
    def __init__(self, root_dir="/muis", data_split="train", image_dir="images", mask_dir="masks",
                 image_ext=".png", mask_ext=".png", image_transforms=ToTensor(), mask_transforms=ToTensor(),
                 color_mapping=None):
        np.random.seed(13)
        image_paths = os.path.join(root_dir, data_split, image_dir, "*{}".format(image_ext))
        self.images = sorted(glob.glob(image_paths))
        self.masks  = [img.replace(image_dir, mask_dir).replace(image_ext, mask_ext) for img in self.images]
        self.image_transforms = image_transforms               # this and below are used for image pre-proc.
        self.mask_transforms = mask_transforms
        self.color_mapping = color_mapping


    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")
        msk = Image.open(self.masks[index]).convert("RGB")
        img = self.image_transforms(img)
        msk = self.mask_transforms(msk)
        msk = self.rgb_to_semantic_mask(msk)
        return img, msk


    def rgb_to_semantic_mask(self, mask):             
        mask  = (mask * 255.0).long()
        mask_flat = mask.view(3, -1).t()
        mapper = torch.tensor(list(self.color_mapping))
        indices = torch.argmax((mask_flat.unsqueeze(1) == mapper.unsqueeze(0)).all(dim=-1).int(), dim=-1)
        mask1D = indices.view(mask.shape[1], mask.shape[2])
        maskND = torch.eye(len(self.color_mapping), dtype=torch.float32)[mask1D].permute(2,0,1)
        return maskND



# Data Transformation
# This is where you can write all your data preprocessing and data augmentation function
# It is always preferable to have different transformation for the training and evaluation sets

mean_imagenet = [0.485, 0.456, 0.406]
std_imagenet  = [0.485, 0.456, 0.406]
base_size = 200
img_size = [224, 224]


train_image_transforms = torchvision.transforms.Compose([
    ToTensor(),
    # CenterCrop(base_size),
    Resize(size=(224,224)),
    Normalize(mean=mean_imagenet, std=std_imagenet),
])

train_mask_transforms = torchvision.transforms.Compose([
    ToTensor(),
    # CenterCrop(base_size),
    Resize(size=(224,224), interpolation=torchvision.transforms.InterpolationMode.NEAREST_EXACT),
])

eval_image_transforms = torchvision.transforms.Compose([
    ToTensor(),
    Resize(size=(224,224)),
    Normalize(mean=mean_imagenet, std=std_imagenet),
])

eval_mask_transforms = torchvision.transforms.Compose([
    ToTensor(),
    Resize(size=(224,224), interpolation=torchvision.transforms.InterpolationMode.NEAREST_EXACT),
])


# Test your dataloader # Be sure your data loader works as desired before using it to train your model

BATCH_SIZE = 46


# Build dataset for different data split
train_dataset = MyDataset(root_dir=filepath, data_split="train", image_dir=image_dir, mask_dir=mask_dir,
                          image_ext=image_ext, mask_ext=mask_ext, image_transforms=train_image_transforms,
                          mask_transforms=train_mask_transforms, color_mapping=color_mapping)

val_dataset = MyDataset(root_dir=filepath, data_split="val", image_dir=image_dir, mask_dir=mask_dir,
                          image_ext=image_ext, mask_ext=mask_ext, image_transforms=eval_image_transforms,
                          mask_transforms=eval_mask_transforms, color_mapping=color_mapping)

test_dataset = MyDataset(root_dir=filepath, data_split="test", image_dir=image_dir, mask_dir=mask_dir,
                          image_ext=image_ext, mask_ext=mask_ext, image_transforms=eval_image_transforms,
                          mask_transforms=eval_mask_transforms, color_mapping=color_mapping)



# Build their loader, include a batch size, data shuffling and any other feature.
train_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = False)
val_dataloader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = True)
test_dataloader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = True)
len(test_dataloader)

# Check one sample
image_i, mask_i = next(iter(val_dataloader))


print("Image shape = {} | Mask shape = {}".format(image_i.shape, mask_i.shape))

# Torch uses the channel-first tensor, we can transpose to channel-last to visualize
image_1 = image_i[0].permute(1,2,0)
mask_1 = mask_i[0].permute(1,2,0)


# Convert semantic mask to singel channel and final to rgb channel to visualize
semantic_mask_1D = nD_to_1D(mask_1)
recovered_rgb_msk = semantic_to_rgb(semantic_mask_1D, color_mapping)

# Plot
display_image(images=[image_1, recovered_rgb_msk],
               titles=['Image', 'Target mask'])


# Define SegResNet model
class SegResNet(nn.Module):
    def __init__(self, num_classes=21):
        super(SegResNet, self).__init__()
        # Charger le modèle ResNet pré-entraîné
        self.backbone = models.resnet50(pretrained=True)

        # Remplacer la dernière couche fully connected par des couches de segmentation
        self.classifier = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, num_classes, kernel_size=1),
            nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)
        )

    def forward(self, x):
        # Utiliser ResNet comme feature extractor
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # Appliquer le classifieur de segmentation
        x = self.classifier(x)

        return x





# Initialize the model (select 2 or 3 classes)
seg_model = SegResNet(num_classes=2) 

# Send the model to the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seg_model = seg_model.to(device)

# Display the model architecture
print(seg_model)

# Check one sample
image_i, gt_mask_i = next(iter(train_dataloader))
input_image = image_i.to(device)

# inference
seg_model.eval()
with torch.no_grad():
    pd_mask = seg_model(input_image)

# Move the prediction to the CPU
pd_mask_i = pd_mask.cpu()

print("Image shape = {} | GT Mask shape = {} | Pred Mask shape = {}".format(image_i.shape, gt_mask_i.shape, pd_mask_i.shape))

# Check one sample
image_i, gt_mask_i = next(iter(train_dataloader))
input_image = image_i.to(device)

# inference
seg_model = seg_model.to(device)
seg_model.eval()
pd_mask = seg_model(input_image)

# Move the masks to the CPU and convert them to numpy arrays.
gt_mask_i = gt_mask_i.cpu()
pd_mask_i = pd_mask.cpu()

# Take the first image from the batch for visualization.
image_1 = image_i[0].permute(1, 2, 0).cpu().numpy()

# Convert the semantic mask to an RGB mask
gt_mask_1 = gt_mask_i.argmax(1)[0].numpy()
pd_mask_1 = pd_mask_i.argmax(1)[0].numpy()

# Display the unique values in the masks.
print("GT Mask unique values:", np.unique(gt_mask_1))
print("Predicted Mask unique values:", np.unique(pd_mask_1))

# Convert the masks to RGB
gt_rgb_msk = semantic_to_rgb(gt_mask_1, color_mapping)
pd_rgb_msk = semantic_to_rgb(pd_mask_1, color_mapping)


fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(image_1)
ax[0].set_title('Input Image')
ax[1].imshow(gt_rgb_msk)
ax[1].set_title('Ground Truth Mask')
ax[2].imshow(pd_rgb_msk)
ax[2].set_title('Predicted Mask')
plt.show()



#Train the model for one epoch
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss




#Calculate Intersection over Union (IoU) between predicted masks and target masks.
def calculate_iou(pred_masks, target_masks):
    intersection = torch.logical_and(pred_masks, target_masks).sum()
    union = torch.logical_or(pred_masks, target_masks).sum()
    iou = (intersection.float() / union.float()).item()
    return iou

#Calculate the accuracy of the predicted masks.
def calculate_accuracy(pred_masks, target_masks):
    correct = (pred_masks == target_masks).sum().item()
    total_pixels = target_masks.numel()
    accuracy = correct / total_pixels
    return accuracy

#Calculate the F1 score between predicted masks and target masks.
def calculate_f1_score(pred_masks, target_masks):
    TP = ((pred_masks == 1) & (target_masks == 1)).sum().item()
    FP = ((pred_masks == 1) & (target_masks == 0)).sum().item()
    FN = ((pred_masks == 0) & (target_masks == 1)).sum().item()

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return f1_score

#Evaluate the model on the validation dataset.
def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    total_iou = 0.0
    total_accuracy = 0.0
    total_f1_score = 0.0  # Nouvelle variable pour stocker le score F1
    num_batches = len(val_loader)

    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item() * images.size(0)

            pred_masks = outputs.argmax(1)
            iou = calculate_iou(pred_masks, masks.argmax(1))
            total_iou += iou

            accuracy = calculate_accuracy(pred_masks, masks.argmax(1))
            total_accuracy += accuracy

            f1_score = calculate_f1_score(pred_masks, masks.argmax(1))  # Calculer le score F1
            total_f1_score += f1_score

    epoch_loss = running_loss / len(val_loader.dataset)
    avg_iou = total_iou / num_batches
    avg_accuracy = total_accuracy / num_batches
    avg_f1_score = total_f1_score / num_batches  # Calculer le score F1 moyen
    return epoch_loss, avg_iou, avg_accuracy, avg_f1_score  # Retourner également le score F1 dans les valeurs de retour

# Train and evaluate the model over multiple epochs.
def train_loop(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=20):
    train_losses = []
    val_losses = []
    val_iou = []
    val_accuracy = []
    val_f1_score = []  # Nouvelle liste pour stocker les scores F1

    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        val_loss, iou, accuracy, f1_score = evaluate_model(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_iou.append(iou)
        val_accuracy.append(accuracy)
        val_f1_score.append(f1_score)  

        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Train Loss: {train_loss:.4f}, '
              f'Val Loss: {val_loss:.4f}, '
              f'Val IoU: {iou:.4f}, '
              f'Val Accuracy: {accuracy:.4f}, '
              f'Val F1 Score: {f1_score:.4f}, '  
              f'Time: {time.time() - start_time:.2f}s')

    return train_losses, val_losses, val_iou, val_accuracy, val_f1_score 


# Define your criterion (loss function) and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(seg_model.parameters(), lr=1e-7)

# Move model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seg_model = seg_model.to(device)

# Train the model (if you load an already trained model, do not run this code).
#num_epochs = 300
#seg_model.eval()
#train_losses, val_losses, val_iou, val_accuracy, val_f1_score = train_loop(seg_model, train_dataloader, val_dataloader, criterion, optimizer, device, num_epochs=num_epochs)

# Plot training and validation metrics
#plt.figure(figsize=(16, 5))  
#plt.subplot(1, 4, 1)  # Ajout d'une sous-figure pour la perte
#plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
#plt.plot(range(1, num_epochs+1), val_losses, label='Val Loss')
#plt.xlabel('Epochs')
#plt.ylabel('Loss')
#plt.legend()
#plt.title('Training and Validation Loss')

#plt.subplot(1, 4, 2)  
#plt.plot(range(1, num_epochs+1), val_iou, label='Val IoU')
#plt.xlabel('Epochs')
#plt.ylabel('IoU')
#plt.legend()
#plt.title('Validation IoU')

#plt.subplot(1, 4, 3) 
#plt.plot(range(1, num_epochs+1), val_accuracy, label='Val Accuracy')
#plt.xlabel('Epochs')
#plt.ylabel('Accuracy')
#plt.legend()
#plt.title('Validation Accuracy')

#plt.subplot(1, 4, 4)  
#plt.plot(range(1, num_epochs+1), val_f1_score, label='Val F1 Score')
#plt.xlabel('Epochs')
#plt.ylabel('F1 Score')
#plt.legend()
#plt.title('Validation F1 Score')

#plt.tight_layout()  
#plt.show()





# Load the trained model (Replace with your own directory).
seg_model.load_state_dict(torch.load('D:/Internship_wemmert/models/segmentation_SegResNet_2cl.pth', map_location=torch.device('cpu')))



# Evaluation on the test set
test_loss, test_iou, test_accuracy, test_f1_score = evaluate_model(seg_model, test_dataloader, criterion, device)

print(f'Test Loss: {test_loss:.4f}, Test IoU: {test_iou:.4f}, Test Accuracy: {test_accuracy:.4f}, Test F1 Score: {test_f1_score:.4f}')



# Visualize the images and prediction

def visualize_preds(model, dataloader, choice, device):
    iterloader = iter(dataloader)
    if choice >= len(iterloader):
        choice = 0
    for _ in range(choice-1):
        next(iterloader)
    model.eval()
    with torch.no_grad():
        inputs, labels = next(iterloader)
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs) # inference
        preds   = outputs.softmax(1).argmax(1)
        targets = labels.softmax(1).argmax(1)
    for img, pred, target in zip(inputs, preds, targets):
        img = img.permute(1,2,0).cpu()
        target = target.cpu()
        pred = pred.cpu()
        print(np.unique(pred))
        gt_rgb_msk = semantic_to_rgb(target, color_mapping)
        pd_rgb_msk = semantic_to_rgb(pred, color_mapping)
        display_image(images=[img.cpu(), gt_rgb_msk, pd_rgb_msk],
                  titles=['Image', 'Target mask', "Predicted mask"])
    return None



iterloader = iter(test_dataloader)
N = len(test_dataloader)
choice = random.choice(list(range(N)))
print("Chosing batch {} out of {} batches".format(choice, N))


visualize_preds(seg_model, test_dataloader, choice, device)




