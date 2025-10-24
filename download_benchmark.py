from huggingface_hub import snapshot_download

## code to download PatentFullBench and SciFullBench datasets
snapshot_download(
    repo_id="Jackson0018/Paper2PaperRetrievalBench",
    repo_type="dataset",
    local_dir=".",
    local_dir_use_symlinks=False,
)