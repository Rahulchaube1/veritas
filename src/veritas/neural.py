"""Local-only quantized transformer inference. No remote model code execution."""

import json
from pathlib import Path

import numpy as np


class Encoder:
    def __init__(self, root: Path, length: int = 192):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.tok = Tokenizer.from_file(str(root / "tokenizer.json"))
        self.tok.enable_truncation(max_length=length)
        config = json.loads((root / "config.json").read_text())
        pad = config.get("pad_token_id", 0)
        self.tok.enable_padding(pad_id=pad, pad_token=self.tok.id_to_token(pad))
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(root / "onnx/model_quantized.onnx"), opts, providers=["CPUExecutionProvider"]
        )
        self.names = {node.name for node in self.session.get_inputs()}
        self.labels = {int(k): v.lower() for k, v in config.get("id2label", {}).items()}

    def run(self, inputs: list, pooled: bool = False) -> np.ndarray:
        output = []
        for start in range(0, len(inputs), 16):
            enc = self.tok.encode_batch(inputs[start : start + 16])
            tensors = {
                k: np.asarray([getattr(x, attr) for x in enc], dtype=np.int64)
                for k, attr in [
                    ("input_ids", "ids"),
                    ("attention_mask", "attention_mask"),
                    ("token_type_ids", "type_ids"),
                ]
                if k in self.names
            }
            raw = self.session.run(None, tensors)[0]
            if pooled:
                mask = tensors["attention_mask"][:, :, None]
                raw = (raw * mask).sum(1) / mask.sum(1).clip(1)
                raw /= np.linalg.norm(raw, axis=1, keepdims=True).clip(1e-9)
            output.append(raw)
        return np.concatenate(output).astype("float32")
