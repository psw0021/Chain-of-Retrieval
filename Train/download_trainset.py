from huggingface_hub import snapshot_download

## code to download Train Set for Policy Model Rollout
snapshot_download(
    repo_id="Jackson0018/Final_Train_Set",
    repo_type="dataset",
    local_dir=".",
    local_dir_use_symlinks=False,
)