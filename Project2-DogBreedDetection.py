#Predict the dog breed using pretrained model VGG16
from PIL import Image
#image classifier project
def process_image(image):
    #https://stackoverflow.com/questions/273946/how-do-i-resize-an-image-using-pil-and-maintain-its-aspect-ratio
    size = 250, 250
    means = np.array([0.485, 0.456, 0.406])
    sd = np.array([0.229, 0.224, 0.225])
    
    im = Image.open(image)
    im.thumbnail(size, Image.ANTIALIAS)
    #https://stackoverflow.com/questions/16646183/crop-an-image-in-the-centre-using-pil
    width, height = im.size   # Get dimensions
    new_width = new_height = 224
    
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    im = im.crop((left, top, right, bottom))
    np_image = np.array(im)/255
    np_image = (np_image - means)/sd
    image_tensor = torch.from_numpy(np_image.transpose()).type(torch.FloatTensor)  
    return image_tensor
	
def VGG16_predict(img_path):
    print(img_path)
    image = process_image(img_path)
    print(image.size())
    imshow(image)
    image = image.unsqueeze(0)
    with torch.no_grad():
        result = VGG16(image)
    softmax = nn.Softmax(dim=1)
    output = softmax(result)
    top_p, top_c = output.topk(1, dim=1)
    top_p = top_p.detach().numpy().tolist()[0][0]
    top_c = top_c.detach().numpy().tolist()[0][0]
    print(top_p)
    print(top_c)
    return top_c # predicted class index
	
def dog_detector(img_path):
    ## TODO: Complete the function.
    output = VGG16_predict(img_path)
    if 151<= output <=268:
        return True
    return False
	
#Load data for training and validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'valid']}
loaders_scratch = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=0)
              for x in ['train', 'valid']}

#Define the CNN model			  
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(28*28*64, 2048)
        self.fc2 = nn.Linear(2048, 133)
        self.dropout = nn.Dropout(0.25)
		
    def forward(self, x):
        #224*224*3
        x = self.pool(F.relu(self.conv1(x)))
        #112*112*16
        x = self.pool(F.relu(self.conv2(x)))
        #56*56*32
        x = self.pool(F.relu(self.conv3(x)))
        #28*28*64
        x = x.view(-1, 28*28*64)
        x = x.self.dropout(x)
        x = F.relu(x.fc1(x))
        x = x.self.dropout(x)
        x = x.fc2(x)
        return x
		
#Define the loss function and optimizer
import torch.optim as optim

riterion_scratch = nn.CrossEntropyLoss()
optimizer_scratch = optim.SGD(model_scratch.parameters(), lr=0.01)
