Here I tried to:
+ extract the bounding boxes form the images with the state of the art model DINO
+ extract from an image the caption using blip1 (blip2 better but too expensive)
+ from the caption extracted with blip1 I encoded them using clip and I computed different distance metrics
+ additionally, I found that we can use blip as a feature extractor given a text description
+ we can implemet our image-text matching architecture using clip (agg for the computational power)
