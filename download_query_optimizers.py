from huggingface_hub import snapshot_download

## choose repo id from the provided models on Hugging Face and set rep_id in below code
## chooose the path to store the downloaded models inside local_dir

snapshot_download(
    repo_id="Jackson0018/Llama-3.2-3B-Instruct_INFV",  
    repo_type="model",                 
    local_dir="Models",              
    local_dir_use_symlinks=False,      
)