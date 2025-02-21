import gradio as gr
from fastapi import FastAPI
import uvicorn

from mikazuki import nibo_process




class LoraTrain:
    process = None

    current_task_uid = 0
    state = "idle"
    return_code = 0

    def __init__(self):
        pass


    def train_task(self,task_uid,task_file):
        self.current_task_uid = task_uid
        self.train(task_file)


    def train(self,toml_file):
        self.process = nibo_process.run_train(toml_file,"./scripts/stable/sdxl_train_network.py")
        self.state = "busy"
        return {"status": "Training started"}

    def check_status(self):
        if self.process == None:
            return {"uuid": self.current_task_uid,"state": self.state ,"return_code": self.return_code}
        ret = self.process.poll()
        if ret == None:
            print("Training in progress")
        else:
            print("Training completed")
            self.process = None
            self.state = "idle"
            self.return_code = ret
        return {"uuid": self.current_task_uid,"state": self.state ,"return_code": self.return_code}



lora_tain = LoraTrain()

app = FastAPI()

@app.get("/")
def read_root():
    return {"Train": "Lora"}

@app.get("/check/")
def api_check():
    return check_status()

@app.post("/train/")
def api_train(toml_file):
    train(toml_file)
    return {"status": "Training started"}

def train(toml_file):
    print(toml_file)
    lora_tain.train(toml_file)
    return "Training started"

def check_status():
    return lora_tain.check_status()

with gr.Blocks(title="Nibo Train") as webapp:
    file_path = gr.Textbox(label="Toml File")
    with gr.Row():
        btn_train = gr.Button("Train",interactive=True)
        btn_queue = gr.Button("Queue",interactive=True)
    output = gr.Textbox(label="Status")

    btn_train.click(train,inputs=[file_path],outputs=[output])
    btn_queue.click(check_status,outputs=[output])

app = gr.mount_gradio_app(app,webapp,path="/gradio")
print(app)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)