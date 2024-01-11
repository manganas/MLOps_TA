from fastapi import FastAPI
from fastapi import UploadFile, File
from fastapi.responses import FileResponse

from typing import Optional
from http import HTTPStatus

from enum import Enum
import re
from pydantic import BaseModel

import cv2

app = FastAPI()


class ItemEnum(Enum):
    alexnet = "alexnet"
    lenet = "lenet"
    resnet = "resnet"


@app.get("/")
def read_root():
    """
    Health check
    """
    message = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return message


@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}


@app.get("/restricted_items/{item_id}")
def read_item(item_id: ItemEnum):
    return {"item_id": item_id}


@app.get("/query_items")
def read_item(item_id: int):
    return {"item_id": item_id}


## database example
database = {"username": [], "password": []}


@app.post("/login/")
def login(username: str, password: int):
    usernames = database["username"]
    passwords = database["password"]

    if username not in usernames and password not in passwords:
        with open("database.txt", "a") as f:
            f.write(f"{username},{password}\n")

        usernames.append(username)
        passwords.append(password)

        return "login saved"

    return HTTPStatus.OK.phrase


@app.get("/text_model/")
def contains_email(data: str):
    regex = r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)"
    re2 = r"@([\w]*)."

    is_email = re.fullmatch(regex, data) is not None
    domain = re.findall(re2, data)

    response = {
        "input": data,
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "email": is_email,
        "domain": domain[0] if domain[0] in ["hotmail", "gmail"] else "unknown",
    }

    return response


class EmailItem(BaseModel):
    email: str
    domain: str


@app.post("/check_domain/")
def check_domain(data: EmailItem):
    regex = r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)"
    # re2 = r"@([\w]*)."

    response = {
        "is_email": re.fullmatch(regex, data.email) is not None,
        "domain_matches": data.domain in ["hotmail", "gmail"],
        "message": HTTPStatus.OK.phrase,
    }

    return response


@app.post("/cv_model/")
async def cv_model(
    data: UploadFile = File(...), h: Optional[int] = 28, w: Optional[int] = 28
):
    with open("image.jpg", "wb") as image:
        content = await data.read()
        image.write(content)
        image.close()

    img = cv2.imread("image.jpg")
    res = cv2.resize(img, (h, w))

    cv2.imwrite("image_red.jpg", res)

    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "input": data,
        "output": FileResponse("image_resize.jpg"),
    }

    return response
