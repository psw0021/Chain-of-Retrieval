from huggingface_hub import snapshot_download

## code to download raw inward citation information obtained using Semantic Scholar API
snapshot_download(
    repo_id="Jackson0018/Raw_Train_Dataset_Semantic_Scholar",
    repo_type="dataset",
    local_dir=".",
    local_dir_use_symlinks=False,
)