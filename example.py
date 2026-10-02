# scGPT cell embedding  example
import numpy as np
from nanomirrors.auto import AutoModel, AutoInputEncoder

genes = ['DUX4L30', 'USP17L16P', 'RPL7P23', 'ZZZ3', 'ZSCAN9', 'VN1R104P']
exprs = np.array([[1.0, 0.0, 23.0, 6.0, 5.0, 4.0], [0.0, 0.0, 3.0, 7.0, 2.0, 1.0]])

model_name = "scGPT_human" # or "Geneformer-V2-104M", "Geneformer-V1-10M"

model = AutoModel.from_pretrained(model_name)
model.eval()

input_encoder = AutoInputEncoder.from_pretrained(model_name, gene_names=genes)
inputs = input_encoder.encode({"exprs": exprs})

embeddings = model.encode_cell(**inputs)
print(embeddings.shape)