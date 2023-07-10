## Usage 


### 1. Download the checkpoint of the model

```bash
gdown 13k2EJy8WTMENrrGPpW-82xSHkY6E0-n9
```

### 2. Download RefCOCOg dataset

```bash
gdown 1tkTUpbdkPqQ1JgHcVlsT0ikjcftF4P6x
unzip refcocog.tar.zip
rm refcocog.tar.zip
tar -xvf refcocog.tar
rm refcocog.tar
```

### 3. Import the module & Check the the pred bbox

```python
# Load refs and annotations
import pickle

with open("./refcocog/annotations/refs(umd).p", "rb") as fp:
refs = pickle.load(fp)

# 'annotations' will be a dict object mapping the 'annotation_id' to the 'bbox' to make search faster
with open("./refcocog/annotations/instances.json", "rb") as fp:
data = json.load(fp)
annotations = dict(sorted({ann["id"]: ann["bbox"] for ann in data["annotations"]}.items()))


# Load the model
sentence_rephrase_module = SentenceRephrase(t5_model, t5_tokenizer, device).to(device)
yolo_detector_module = YOLOv8DetectorModule(model_yolo, clip_preprocess, device).to(device)

fine_tuned_model = FinetunedRephrasing().to(device)
fine_tuned_model = fine_tuned_model.load_from_checkpoint('./c2lip-box_simpified_128-epoch=70-val_loss=0.00-other_metric=0.00.ckpt')

model_finetuned = RephrasingDetectorModel(clip_model, clip.tokenize, yolo_detector_module, sentence_rephrase_module, fine_tuned_model, device)

# count the number of parameters
print(f"Number of parameters: {sum(p.numel() for p in model_finetuned.parameters() if p.requires_grad)/1000000} M")

dataset = RefCOCOg(refs, annotations, split="test")
# dataset = RefCOCOg(refs, annotations, split="val")

interval = np.arange(30, 3000, 30)

path_imgs = [[dataset[i]['file_name']] for i in interval]
captions = [[dataset[i]['caption']] for i in interval]


bboxes_pred = model_finetuned(path_imgs, captions).detach().cpu()

# plot the images with the predicted bounding boxes
images = [load_image_pil(i[0]) for i in path_imgs]
yolo_detector_module.plot_image_bbox(bboxes_pred, images, captions)
```