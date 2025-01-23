import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from PIL import Image

from dataset_tag_editor.tagger import WaifuDiffusion

tag = WaifuDiffusion("wd-v1-4-convnext-tagger-v2",0.7)
tag.start()

image = Image.open("D:/datasets/1.jpg")

output = tag.predict(image,0.7)
print(output)