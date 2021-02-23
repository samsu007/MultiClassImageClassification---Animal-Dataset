from flask import Flask, render_template, request
import torch
from torchvision import transforms
from PIL import Image

app = Flask(__name__)

model_path = "D:\pytorch\CalTech256Classification\\animal_5_resnet50_model.pt"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/", methods=['POST'])
def predict():
    f = request.files['image']

    model = torch.load(model_path)

    tranform = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    image = Image.open(f)

    image_tensor = tranform(image)

    if torch.cuda.is_available():
        image_tensor = image_tensor.view(1, 3, 224, 224).cuda()
    else:
        image_tensor = image_tensor.view(1, 3, 224, 224)

    with torch.no_grad():
        model.eval()

        out = model(image_tensor)
        ps = torch.exp(out)

        topk, topclass = ps.topk(1, dim=1)

        option = {'cane': 0, 'cavallo': 1, 'elefante': 2, 'farfalla': 3, 'gallina': 4}

        idx_to_class = {v: k for k, v in option.items()}
        print(idx_to_class)

        cls = idx_to_class[topclass.cpu().numpy()[0][0]]
        score = topk.cpu().numpy()[0][0]

        print(cls, score)

    return render_template("index.html", cls=cls, score=score)
