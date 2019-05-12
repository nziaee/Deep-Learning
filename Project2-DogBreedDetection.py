#Step1-Use pretrained model

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

#Step2-Define the model from scratch	
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
        self.fc1 = nn.Linear(28*28*64, 500)
        self.fc2 = nn.Linear(500, 133)
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

#Train Model steps

	# clear the gradients of all optimized variables
    optimizer.zero_grad()
	# forward pass: compute predicted outputs by passing inputs to the model
    output = model(data)
	# calculate the batch loss
    loss = criterion(output, target)
    # backward pass: compute gradient of the loss with respect to model parameters
    loss.backward()
    # perform a single optimization step (parameter update)
    optimizer.step()
    train_loss += loss.item()*data.size(0)
	
#Validate the model
	output = model(data)
    loss = criterion(output, target)
    valid_loss += loss.item()*data.size(0)
    _, preds = torch.max(output, 1)  
    running_corrects += torch.sum(preds == target.data)
	
# calculate average losses
    train_loss = train_loss/len(loaders['train'].dataset)
    valid_loss = valid_loss/len(loaders['valid'].dataset)
    # print training/validation statistics
    accuracy = running_corrects.double() / len(loaders['valid'].dataset)
	
#save the model if the validation loss has improved
	if valid_loss < valid_loss_min:
          print('save model.')
          valid_loss_min = valid_loss
          path = F"/content/gdrive/My Drive/{save_path}"
          torch.save({
           'epoch': n_epochs,
           'model_state_dict': model.state_dict(),
           'optimizer_state_dict': optimizer.state_dict(),
           }, path)

#Step3-Transfer learning
import torchvision.models as models
import torch.nn as nn
model_transfer = models.densenet161(pretrained=True)

#Freeze the features
for name, child in model_transfer.named_children():
   if name in ['classifier']:
       for param in child.parameters():
          param.requires_grad = True
   else:
       print(name + ' is frozen')
       for param in child.parameters():
           param.requires_grad = False
		   
#Replace the classifiernum_ftrs = model_transfer.classifier.in_features
print(num_ftrs)
model_transfer.classifier = nn.Linear(num_ftrs, 133)
print(model_transfer.classifier)

#Define the loss function and optimizer
import torch.optim as optim

criterion_transfer = nn.CrossEntropyLoss()
optimizer_transfer = optim.SGD(model_transfer.parameters(), lr=0.01)

def predict_breed_transfer(img_path, model):
    image = process_image(img_path)
    imshow(image)
    image = image.unsqueeze(0)
    with torch.no_grad():
        model = model.cpu()
        result = model(image)
    softmax = nn.Softmax(dim=1)
    output = softmax(result)
    top_p, top_c = output.topk(5, dim=1)
    #print(top_p)
    #print(top_c)
    top_c = top_c.detach().numpy().tolist()[0]
    names = []
    for index in top_c:
      names.append(class_names[index-1])
    #print(top_p)
    print(top_c)
    return names # predicted class index

#Run a test	
model_save_name = 'model_transfer.pt'
path = F"/content/gdrive/My Drive/{model_save_name}"
checkpoint = torch.load(path)
model_transfer.load_state_dict(checkpoint['model_state_dict'])

print(predict_breed_transfer('/content/data/dogImages/train/055.Curly-coated_retriever/Curly-coated_retriever_03868.jpg', model_transfer))

def run_app(img_path):
    if face_detector(img_path):
      print('face')
      print(predict_breed_transfer(img_path, model_transfer))
    elif dog_detector(img_path):
      print('dog')
      print(predict_breed_transfer(img_path, model_transfer))
    else:
      print('Neither human or dog was detected.')
      image = process_image(img_path)
      imshow(image)
