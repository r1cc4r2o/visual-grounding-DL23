import torch
from PIL import Image
import requests
from lavis.models import load_model_and_preprocess
import matplotlib.pyplot as plt

img_url = 'https://www.thehappycatsite.com/wp-content/uploads/2019/05/Eqyptian-cat-names-HC-long.jpg' 
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')   
plt.imshow(raw_image.resize((596, 437)))
plt.axis('off')
plt.show()

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_opt", model_type="pretrain_opt2.7b", is_eval=True, device=device
)

image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

print(model.generate({"image": image}))