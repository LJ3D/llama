from typing import Tuple
import os
import sys
import torch
#import torch_directml
import fire
import time
import json

from pathlib import Path

from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama import ModelArgs, Transformer, Tokenizer, LLaMA

from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QTextEdit, QVBoxLayout, QHBoxLayout, QPushButton


class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Text Input Window")
        self.setGeometry(100, 100, 500, 300)

        # Create widgets
        self.input_label = QLabel("Input Text:")
        self.input_text = QTextEdit()
        self.output_label = QLabel("Output Text:")
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)  # Set the output text to read-only
        self.submit_button = QPushButton("Submit")
        self.submit_button.clicked.connect(self.submit_text)

        # Create layouts
        input_layout = QVBoxLayout()
        input_layout.addWidget(self.input_label)
        input_layout.addWidget(self.input_text)

        output_layout = QVBoxLayout()
        output_layout.addWidget(self.output_label)
        output_layout.addWidget(self.output_text)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.submit_button)

        main_layout = QVBoxLayout()
        main_layout.addLayout(input_layout)
        main_layout.addLayout(output_layout)
        main_layout.addLayout(button_layout)

        # Set the main layout of the window
        self.setLayout(main_layout)

        # Set up the pytorch model

    def submit_text(self):
        text = self.input_text.toPlainText()
        self.output_text.append(text)
        self.input_text.clear()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())
