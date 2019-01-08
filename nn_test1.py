import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
import torchvision.datasets
from PIL import Image
# from bokeh.plotting import figure
# from bokeh.io import show
# from bokeh.models import LinearAxis, Range1d
import numpy as np
import os

print(os.getcwd())
cwd=os.getcwd()



num_epochs = 6
num_classes = 7
batch_size = 30
learning_rate = 0.001

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#torch_loader=torchvision.datasets.ImageFolder((cwd+'/data/'))
#train_loader=torch.utils.data.DataLoader(torch_loader, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None)
#test_loader=torch.utils.data.DataLoader(cwd+'/test/', batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None)


# class CustomDatasetFromCSV(Dataset):#train_loader
#     def __init__(self, csv_path, height, width, transforms=None):
#         """
#         Args:
#             csv_path (string): path to csv file
#             height (int): image height
#             width (int): image width
#             transform: pytorch transforms for transforms and tensor conversion
#         """
#         self.data = pd.read_csv(csv_path)
#         self.labels = np.asarray(self.data.iloc[:, 0])
#         self.height = height
#         self.width = width
#         self.transforms = transform
#
#     def __getitem__(self, index):
#         single_image_label = self.labels[index]
#         # Read each 784 pixels and reshape the 1D array ([784]) to 2D array ([28,28])
#         img_as_np = np.asarray(self.data.iloc[index][1:]).reshape(600, 450).astype('uint8')
#         # Convert image from numpy array to PIL image, mode 'L' is for grayscale
#         img_as_img = Image.fromarray(img_as_np)
#         img_as_img = img_as_img.convert('L')
#         # Transform image to tensor
#         if self.transforms is not None:
#             img_as_tensor = self.transforms(img_as_img)
#         # Return image and the label
#         return (img_as_tensor, single_image_label)
#
#     def __len__(self):
#         return len(self.data.index)

class CustomDatasetFromImages(Dataset):
    def __init__(self, csv_path):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Transforms
        self.to_tensor = transforms.ToTensor()
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=1)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 1])
        # Second column is the labels
        self.label_arr = np.asarray(self.data_info.iloc[:, 2])
        # Third column is for an operation indicator
        self.operation_arr = np.asarray(self.data_info.iloc[:, 2])
        # Calculate len
        self.data_len = len(self.data_info.index)

    def __getitem__(self, index):
        cwd=os.getcwd()
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]
        # Open image
        print('cwd:', os.getcwd())


        img_as_img = Image.open(cwd+'/data/bilder/'+single_image_name)

        # Check if there is an operation
        some_operation = self.operation_arr[index]
        # If there is an operation
        if some_operation:
            # Do some operation on image
            # ...
            # ...
            pass
        # Transform image to tensor
        img_as_tensor = self.to_tensor(img_as_img)

        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]

        return (img_as_tensor, single_image_label)

    def __len__(self):
        return self.data_len

if __name__ == "__main__":
    transformations = transforms.Compose([transforms.ToTensor()])

    #trainloader
    train_loader = \
        CustomDatasetFromImages('data/bilder/HAM10000_metadata.csv')
    #testloader
    #test_loader = CustomDatasetFromCSV('test/HAM10000_metadata.csv', 600, 450, transformations)
    print('datengeladen')



class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(7 * 7 * 64, 1000)
        self.fc2 = nn.Linear(1000, 10)

def forward(self, x):
    out = self.layer1(x)
    out = self.layer2(out)
    out = out.reshape(out.size(0), -1)
    out = self.drop_out(out)
    out = self.fc1(out)
    out = self.fc2(out)
    return out

model = ConvNet()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
loss_list = []
acc_list = []
#os.chdir('data/bilder/')
for epoch in range(num_epochs):
    print('cwd:', os.getcwd())
    for i, (images, labels) in enumerate(train_loader):
        print(images)
        # Run the forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())

        # Backprop and perform Adam optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track the accuracy
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        acc_list.append(correct / total)

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                          (correct / total) * 100))


# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format((correct / total) * 100))

# Save the model and plot
torch.save(model.state_dict(), MODEL_STORE_PATH + 'conv_net_model.ckpt')











#face_dataset = FaceLandmarksDataset(csv_file='HAM10000_metadata.csv',


# class FaceLandmarksDataset(Dataset):
#     """Face Landmarks dataset."""
#
#     def __init__(self, csv_file, root_dir, transform=None):
#         print('test')
#         """
#         Args:
#             csv_file (string): Path to the csv file with annotations.
#             root_dir (string): Directory with all the images.
#             transform (callable, optional): Optional transform to be applied
#                 on a sample.
#         """
#         self.landmarks_frame = pd.read_csv('/data/HAM10000_metadata.csv')
#         self.root_dir = ('/data/')
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.landmarks_frame)
#
#     def __getitem__(self, idx):
#         img_name = os.path.join(self.root_dir,
#                                 self.landmarks_frame.iloc[idx, 0])
#         image = io.imread(img_name)
#         landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
#         landmarks = landmarks.astype('float').reshape(-1, 2)
#         sample = {'image': image, 'landmarks': landmarks}
#
#         if self.transform:
#             sample = self.transform(sample)
#
#         return sample