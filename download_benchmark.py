from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Jackson0018/Paper2PaperRetrievalBench",
    repo_type="dataset",
    local_dir=".",
    local_dir_use_symlinks=False,
)