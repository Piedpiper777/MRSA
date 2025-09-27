from modelscope import snapshot_download

model_id = "google-bert/bert-base-chinese"    
model_dir = r"/workspace/MRSA/models"

model_dir = snapshot_download(model_id, cache_dir=model_dir, revision="master")