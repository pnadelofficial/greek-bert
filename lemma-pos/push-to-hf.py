from huggingface_hub import HfApi

api = HfApi()
api.create_repo("pnadel/ancient-greek-morph-tagger", private=False)

api.upload_folder(
    folder_path="./greek-lemma-pos-model",
    repo_id="pnadel/ancient-greek-morph-tagger",
)

api.upload_file(
    path_or_fileobj="train.py",
    path_in_repo="train.py",
    repo_id="pnadel/ancient-greek-morph-tagger",
)