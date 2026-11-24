# Model cards

| File or model | Training and objective | Scope |
|---|---|---|
| ranker.json | Regularized logistic regression on official SciFact train claims and sampled documents. Inputs are title and body token coverage and Jaccard overlap. | Optional rank adjustment. Random negatives are often easier than retrieved negatives. See training pair counts and validation losses in results/ranker_training.json. |
| rationale_gate.json | Binary logistic regression on official train rationales and nearby unannotated sentences, including hard negatives from BM25. Select threshold by F₀.₅ on disjoint validation abstracts. IDF and seven coefficients exported as JSON. | Passage eligibility in the neural scientific configuration. Incomplete rationales introduce label noise. |
| synthetic_calibration.json | Least-squares isotonic fit on decisive answers with known truth from 360 designated synthetic calibration worlds. | Estimated agreement with the template generator’s truth conditional on answering. Disabled in normal configurations. |
| Multilingual MiniLM | Upstream sentence embeddings, quantized ONNX, masked mean pooling, HNSW index. | Cross-language retrieval. Multilingual entailment requires separate evaluation. [Model](https://huggingface.co/Xenova/paraphrase-multilingual-MiniLM-L12-v2) |
| MS MARCO MiniLM | Upstream passage reranker, quantized ONNX. | English query–passage relevance. [Model](https://huggingface.co/Xenova/ms-marco-MiniLM-L-6-v2) |
| DeBERTa v3 xsmall | Upstream NLI classifier, quantized ONNX. | English passage-to-claim inference. A maximum softmax score is a stance strength and remains uncalibrated. [Model](https://huggingface.co/Xenova/nli-deberta-v3-xsmall) |

The fetch script pins immutable Hugging Face revisions. Weights stay in the local cache under their upstream terms. Only the three project-trained JSON checkpoints are distributed here. SciFact annotation provenance and license information are in the dataset card. Full-batch CPU fits finish in one optimizer invocation. Resumable minibatch and foundation-model training are outside this release.
