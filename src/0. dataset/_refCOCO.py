from torch.utils.data import Dataset
import json
import os

class RefCOCOg(Dataset):
    """
    Args:
        dataset: a list of dictionaries containing:
        {
            'file_name': # path of the image, images will be loaded on the fly
            'caption': # referring caption
            'ann_id': # annotation ID (one per caption), taken from 'file_name'
            'bbox': # coordinates (xmin, ymin, xmax, ymax) of the bounding box
        }
    """
    def __init__(self, refs, annotations, split="train"):

        self.dataset = [{"file_name": os.path.join("./refcocog/images/", f'{"_".join(elem["file_name"].split("_")[:3])}.jpg'),
                            "caption": elem["sentences"][0]["raw"],
                            "ann_id": int(elem["file_name"].split("_")[3][:-4]),
                            "bbox": annotations[int(elem["file_name"].split("_")[3][:-4])]}
                        for elem in [d for d in refs if d["split"]==split]]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __call__(self, idx):
        print(json.dumps(self.dataset[idx], indent=4))
        
        
# Usage:
# # Load refs and annotations
# import pickle

# with open("./refcocog/annotations/refs(umd).p", "rb") as fp:
# refs = pickle.load(fp)

# # 'annotations' will be a dict object mapping the 'annotation_id' to the 'bbox' to make search faster
# with open("./refcocog/annotations/instances.json", "rb") as fp:
# data = json.load(fp)
# annotations = dict(sorted({ann["id"]: ann["bbox"] for ann in data["annotations"]}.items()))
# dataset = RefCOCOg(refs, annotations, split="test")