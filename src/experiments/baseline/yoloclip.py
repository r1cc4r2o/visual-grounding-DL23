import torch
from CLIP import clip
from PIL import Image, ImageDraw
import cv2
import numpy as np

### Detect BBoxes

# Models
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Images
img = 'zidane.jpg'

# Query
q = "A photo of a little tie"

# Inference
results = yolo_model(img)

# Results
# results.print()
results.show()
#      xmin    ymin    xmax   ymax  confidence  class    name


### Choose best BBox

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

text = clip.tokenize(q).to(device)

max_prob = 0
for bbox in results.xyxy[0].cpu().numpy():
    temp = cv2.imread(img)
    image = np.zeros((temp.shape[0], temp.shape[1], temp.shape[2]), dtype=np.uint8)
    image[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] = temp[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
    image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        
        logits_per_image, logits_per_text = model(image, text)
        prob = logits_per_text.cpu().numpy()[0][0]

        if prob > max_prob:
            max_prob = prob
            best_bbox = bbox

source_img = Image.open(img).convert("RGBA")
draw = ImageDraw.Draw(source_img)
draw.rectangle(((best_bbox[0], best_bbox[1]), (best_bbox[2], best_bbox[3])), width=3)
source_img.save("test.png", "PNG")