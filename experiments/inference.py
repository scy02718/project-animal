import pytorch_lightning as pl

from task import *
from dataset import *

def main():
    image_size = 256
    image_crop = (312, 256)

    device = torch.device("cuda")
    transform = transforms.Compose([
                transforms.Resize(image_size),
                # transforms.CenterCrop(image_crop),
                transforms.ToTensor(),
                transforms.Grayscale(num_output_channels=3),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    model_name = "C:/Users/scy02/projects/project-animal/ckpt/epoch=4-val_acc=0.9897.ckpt"
    inference_model = AnimalBackBone.load_from_checkpoint(model_name)    
    
    img_path = "C:/Users/scy02/projects/project-animal/experiments/inference/Fox2.jpg"
    image = PIL.Image.open(img_path)
    image = transform(image)
    plt.imshow(image.numpy().transpose(1,2,0))
    plt.show()
    image = image.unsqueeze(0)    

    image = image.to(device)

    inference_model.to(device)
    inference_model.eval()
    with torch.no_grad():
        prediction = inference_model(image)
    
    softmax = nn.Softmax(dim = 1)
    output = softmax(prediction)
    print(output)

    classes = ["cat", "cheetah", "dog", "fox", "leopard", "lion", "tiger", "wolf"]
    # classes = ["cat", "dog", "fox"]
    print(classes[torch.argmax(output)])

if __name__=="__main__":
    main()