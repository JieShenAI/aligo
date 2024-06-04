import os
from aligo import Aligo
from transformers import AutoModel


ali_huggingface_home = ".cache/huggingface/hub/"
# local_huggingface_home = "/home/jie/.cache/huggingface/hub/"
colab_huggingface_home = "/root/.cache/huggingface/hub/"

ali = Aligo()


def get_huggingface_model_name(model_name):
    p = os.path.split(model_name)
    if p[0]:
        model_name = "--".join(p)
    return "models--" + model_name


def upload_huggingface_model(model_name):
    # upload_huggingface_model("openai/clip-vit-large-patch14")
    model_name = get_huggingface_model_name(model_name)
    
    # upload lock
    ali.upload_folder(
        folder_path=os.path.join(colab_huggingface_home, ".locks", model_name),
        parent_file_id=ali.get_folder_by_path(
            os.path.join(ali_huggingface_home, ".locks")
        ).file_id,
    )

    # upload model
    ali.upload_folder(
        folder_path=os.path.join(colab_huggingface_home, model_name),
        parent_file_id=ali.get_folder_by_path(
            os.path.join(ali_huggingface_home)
        ).file_id,
    )


def down_huggingface_model(model_name, username="root"):
    if username == "root":
        huggingface_local_home = "/root/.cache/huggingface/hub/"
    else:
        huggingface_local_home = f"/home/{username}/.cache/huggingface/hub/"

    model_name = get_huggingface_model_name(model_name)

    model_id = ali.get_folder_by_path(
            os.path.join(ali_huggingface_home, model_name)
        ).file_id

    lock_id = ali.get_folder_by_path(
            os.path.join(ali_huggingface_home, ".locks", model_name)
        ).file_id

    # model
    local_model_folder = os.path.join(huggingface_local_home)
    if not os.path.exists(local_model_folder):
        os.makedirs(local_model_folder, exist_ok=True)
    ali.download_folder(
                folder_file_id=model_id,
                local_folder=local_model_folder
            )
    
    # lock
    local_lock_folder = os.path.join(huggingface_local_home, ".locks")
    if not os.path.exists(local_lock_folder):
        os.makedirs(local_lock_folder, exist_ok=True)
    ali.download_folder(
                folder_file_id=lock_id,
                local_folder=local_model_folder
            )

    
    

if __name__ == "__main__":
    down_huggingface_model("openai/clip-vit-large-patch14", username="jie")
