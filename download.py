from typing import Literal, get_args
import requests

DRIVE_URL = "https://drive.google.com/drive/folders/1lkHY6Jx4KRqzWliGmMVOl8cM8XGWhpis"

_LANGUAGES = Literal["Arabic", "Czech", "Dutch", "English", "Finnish", "French", "Russian", "Turkish"]
_ARCHITECTURES = Literal["UDPipe2", "UDIFY", "DogTag"]

_LINKS = {
    "https://drive.google.com/file/d/1jJDYUeA1OPuRFa_hYb5SJYzXj7IuIG-B",
}

def get_pipeline(language: _LANGUAGES, architecture: _ARCHITECTURES, multi: bool = True):

    def _get_response(file_name):

        full_url = f"{DRIVE_URL}/{file_name}"

        print(f"Trying: {full_url}")

        response = requests.get(full_url)

        if response.status_code == 400:
            print(f'File not found.')

            return None

        else:
            print('File found.')

            return response

    assert language in get_args(_LANGUAGES), f"No pipeline for '{language}' exists yet. Implemented languages are {get_args(_TYPES)}. You can try training your own using 'train_tagger.py'"

    assert architecture in get_args(_ARCHITECTURES), f"Architecture '{architecture}' not implemented. Implemented architectures are {get_args(_ARCHITECTURES)}."

    file_name = lambda m: f"{architecture}{m}_{language}_merge.ckpt"
    
    

    response = _get_response(file_name('_multi' if multi else '_mono'))

    if response is not None:
        return response

    print(f"Falling back on multi: {not multi}")
    response = _get_response(file_name('_multi' if not multi else '_mono'))

    if response is not None:
        return response

    print(f"Falling back on multi: None")
    response = _get_response(file_name(""))

    if response is not None:
        return response

    else:
        raise FileNotFoundError("File not found for some reason.")

if __name__ == "__main__":
    print(get_pipeline(language="Dutch", architecture="UDPipe2", multi=True).status_code)
