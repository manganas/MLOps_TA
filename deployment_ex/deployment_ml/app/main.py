from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch
from PIL import Image
from http import HTTPStatus

from fastapi import FastAPI
from fastapi import File, UploadFile
from typing import Optional
from pydantic import BaseModel

app = FastAPI()


class Kwargs(BaseModel):
    max_length: Optional[int] = 16
    num_beams: Optional[int] = 8
    num_return_sequences: Optional[int] = 1


model = VisionEncoderDecoderModel.from_pretrained(
    "nlpconnect/vit-gpt2-image-captioning"
)
feature_extractor = ViTFeatureExtractor.from_pretrained(
    "nlpconnect/vit-gpt2-image-captioning"
)
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


@app.get("/")
def get_root():
    return {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }


# @app.post("/predict/")
# def predict_step(input: Kwargs):
#     gen_kwargs = {
#         "max_length": input.max_length,
#         "num_beams": input.num_beams,
#         "num_return_sequences": input.num_return_sequences,
#     }
#     images = []
#     for image_path in input.image_paths:
#         i_image = Image.open(image_path)
#         if i_image.mode != "RGB":
#             i_image = i_image.convert(mode="RGB")

#         images.append(i_image)

#     pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
#     pixel_values = pixel_values.to(device)
#     output_ids = model.generate(pixel_values, **gen_kwargs)
#     preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
#     preds = [pred.strip() for pred in preds]
#     return preds


@app.post("/predict/")
async def predict_step(image: UploadFile = File(...)):
    # gen_kwargs = {
    #     "max_length": input.max_length,
    #     "num_beams": input.num_beams,
    #     "num_return_sequences": input.num_return_sequences,
    # }
    gen_kwargs = {
        "max_length": 16,
        "num_beams": 8,
        "num_return_sequences": 1,
    }

    with open("image.jpg", "wb") as f:
        f.write(image.file.read())

    i_image = Image.open("image.jpg")
    if i_image.mode != "RGB":
        i_image = i_image.convert(mode="RGB")

    pixel_values = feature_extractor(images=i_image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    output_ids = model.generate(pixel_values, **gen_kwargs)
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds
