import requests


def main():
    pload = {"username": "Olivia", "password": "123"}
    response = requests.post("https://httpbin.org/post", data=pload)

    if response.status_code != 200:
        print(response.status_code)

    print(response.text)


if __name__ == "__main__":
    main()
