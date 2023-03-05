# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
import os
import sys
import torch
import fire
import time
import json
import threading

from pathlib import Path

from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama import ModelArgs, Transformer, Tokenizer, LLaMA

from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QTextEdit, QVBoxLayout, QHBoxLayout, QPushButton, QDoubleSpinBox, QSpinBox
from PyQt5.QtCore import QObject, pyqtSignal, QThread


import qdarkstyle

driver = "cuda"

def setup_model_parallel() -> Tuple[int, int]:
    global driver
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    torch.distributed.init_process_group("gloo")
    initialize_model_parallel(world_size)
    if driver=="cuda":
        torch.cuda.set_device(local_rank)
    if driver=="dml":
        pass

    # seed must be the same in all processes
    torch.manual_seed(123)
    return local_rank, world_size


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    global driver
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading..")
    with torch.no_grad():
        checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    if driver=="cuda":
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
    if driver=="cpu" or driver=="dml":
        torch.set_default_tensor_type(torch.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)
    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator



class llamaWorker(QObject):
    finished = pyqtSignal(str)

    def __init__(self, generator, temperature, max_gen_len, top_p):
        super().__init__()
        self.generator = generator
        self.temperature = temperature
        self.max_gen_len = max_gen_len
        self.top_p = top_p

    def setInputText(self, inputText):
        self.inputText = inputText

    def run(self):
        results = self.generator.generate([self.inputText], max_gen_len=self.max_gen_len, temperature=self.temperature, top_p=self.top_p)
        final = ""
        for result in results:
            final += result
        self.finished.emit(final)

class llamaWindow(QWidget):
    def __init__(self, ckpt_dir, tokenizer_path, temperature, max_gen_len, top_p, max_seq_len, max_batch_size):
        super().__init__()
        self.setWindowTitle("LLaMA GUI")
        self.setGeometry(100, 100, 500, 300)

        # Create widgets
        self.textEdit = QTextEdit()
        self.generateButton = QPushButton("Start generating")
        self.generateButton.clicked.connect(self.toggleGenerating)
        self.temperatureLabel = QLabel("Temperature:")
        self.temperatureInput = QDoubleSpinBox(self)
        self.temperatureInput.setValue(temperature)
        self.temperatureInput.setRange(0, 1)
        self.temperatureInput.setSingleStep(0.05)
        self.temperatureInput.valueChanged.connect(lambda: self.changeTemp())
        self.top_pLabel = QLabel("Top_p:")
        self.top_pInput = QDoubleSpinBox(self)
        self.top_pInput.setValue(top_p)
        self.top_pInput.setRange(0, 1)
        self.top_pInput.setSingleStep(0.05)
        self.top_pInput.valueChanged.connect(lambda: self.changeTopP())
        self.max_gen_lenLabel = QLabel("Max chunk generation length (longer is a bit more efficient (i think)):")
        self.max_gen_lenInput = QSpinBox(self)
        self.max_gen_lenInput.setValue(max_gen_len)
        self.max_gen_lenInput.setRange(1, 4096)
        self.max_gen_lenInput.setSingleStep(4)
        self.max_gen_lenInput.valueChanged.connect(lambda: self.changeMaxGenLen())
        
        # Create layouts
        input_layout = QVBoxLayout()
        input_layout.addWidget(self.textEdit)
        input_layout.addWidget(self.temperatureLabel)
        input_layout.addWidget(self.temperatureInput)
        input_layout.addWidget(self.top_pLabel)
        input_layout.addWidget(self.top_pInput)
        input_layout.addWidget(self.max_gen_lenLabel)
        input_layout.addWidget(self.max_gen_lenInput)
        
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.generateButton)

        main_layout = QVBoxLayout()
        main_layout.addLayout(input_layout)
        main_layout.addLayout(button_layout)

        # Set the main layout of the window
        self.setLayout(main_layout)

        # Set up the pytorch model
        local_rank, world_size = setup_model_parallel()
        self.generator = load(ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size)
        self.temperature = temperature
        self.max_gen_len = max_gen_len
        self.top_p = top_p
        self.threadAvailable = True
        self.generating = False
    
    def changeTemp(self):
        self.temperature = self.temperatureInput.value()

    def changeTopP(self):
        self.top_p = self.top_pInput.value()

    def changeMaxGenLen(self):
        self.max_gen_len = self.max_gen_lenInput.value()

    def toggleGenerating(self):
        self.generating = not self.generating
        if self.generating:
            self.generateButton.setText("Stop generating")
            self.submitText()
        else:
            self.generateButton.setText("Start generating")

    def submitText(self):
        if not self.threadAvailable:
            return
        self.threadAvailable = False
        self.textEdit.setReadOnly(True)
        inputText = self.textEdit.toPlainText()
        self.worker = llamaWorker(self.generator, self.temperature, self.max_gen_len, self.top_p)
        self.worker.setInputText(inputText)
        self.thread = threading.Thread(target=self.worker.run)
        self.worker.finished.connect(self.setText)
        self.thread.start()

    def setText(self, text):
        self.threadAvailable = True
        self.textEdit.setReadOnly(False)
        self.textEdit.setText(text)
        if self.generating:
            self.submitText()


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.8,
    max_gen_len = 8,
    top_p: float = 0.95,
    max_seq_len: int = 2048,
    max_batch_size: int = 1,
):
    dark_stylesheet = qdarkstyle.load_stylesheet_pyqt5()
    app = QApplication(sys.argv)
    app.setStyleSheet(dark_stylesheet)
    window = llamaWindow(ckpt_dir, tokenizer_path, temperature, max_gen_len, top_p, max_seq_len, max_batch_size)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    fire.Fire(main)
