from fastapi import FastAPI
from mikazuki import nibo_process

app = FastAPI()

@app.get("/")
def read_root():
    return {"Train": "Lora"}

@app.post("/train/{toml_file}")
def train(toml_file):
    nibo_process.run_train(toml_file,"./scripts/stable/sdxl_train_network.py")
    return {"status": "Training started"}

app.run(host="0.0.0.0", port=8000)