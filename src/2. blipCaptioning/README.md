## Usage 


### 1. Download RefCOCOg dataset

```bash
gdown 1tkTUpbdkPqQ1JgHcVlsT0ikjcftF4P6x
unzip refcocog.tar.zip
rm refcocog.tar.zip
tar -xvf refcocog.tar
rm refcocog.tar
```

### 2. To run the blip evaluation

```python
# Load refs and annotations
with open("./refcocog/annotations/refs(umd).p", "rb") as fp:
  refs = pickle.load(fp)

# 'annotations' will be a dict object mapping the 'annotation_id' to the 'bbox' to make search faster
with open("./refcocog/annotations/instances.json", "rb") as fp:
  data = json.load(fp)
  annotations = dict(sorted({ann["id"]: ann["bbox"] for ann in data["annotations"]}.items()))

# create dataset and dataloader
dataset = RefCOCOg(refs, annotations, split="test")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
yolo_clip_blip = YoloClipBlip(device)

yolo_clip_blip.evaluation(dataset)
```

### 3. To run the blip evaluation with the precomputed predictions

Check out the colab notebook available in the README.md of the root directory. Also, the precomputed predictions are available. The instructions to download them and run the evaluation are in the README.md of the root directory.