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