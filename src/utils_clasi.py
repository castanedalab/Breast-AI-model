import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from model_lightning_clasi import MyModelMulticlass
import pandas as pd
from addict import Dict
from collections import Counter
import onnxruntime as ort

def predict_with_model_onnx(session, dataloader, device):
    predictions = []
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    for batch in dataloader:
        images = batch[0].numpy().astype(np.float32)
        ort_inputs = {input_name: images}
        ort_outs = session.run([output_name], ort_inputs)
        predictions.append(ort_outs[0])
    return np.vstack(predictions)

def load_model_onnx(onnx_path, model_name=None):
    providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(onnx_path, providers=providers)
    return session, "cpu"

def load_model(ckpt_path, model_name="resnet"):
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device("cpu")
    model_opts = Dict({'name': model_name})
    train_par   = Dict({'eval_threshold': 0.5, 'loss_opts': {'name': 'CrossEntropyLoss'}})
    model = MyModelMulticlass(model_opts=model_opts, train_par=train_par)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()
    return model, device

def predict_with_model(model, dataloader, device):
    predictions = []
    with torch.no_grad():
        for images, _, _ in dataloader:
            images = images.to(device)
            preds = model(images)
            probs = torch.nn.functional.softmax(preds, dim=-1)
            predictions.append(probs.cpu().numpy())
    return np.vstack(predictions)

def ensemble_predictions(models, dataloader, device, method="average"):
    label_map = {0:"No follow up",1:"Follow up",2:"Biopsy"}
    models = [m.to(device).eval() for m in models]
    patient_preds = {}
    with torch.no_grad():
        for images, labels, patient_ids in dataloader:
            images = images.to(device)
            model_outs = [torch.nn.functional.softmax(m(images), dim=-1) for m in models]
            if method=="average":
                preds = torch.stack(model_outs).mean(dim=0)
            else:
                raise ValueError("Método no soportado")
            best = torch.argmax(preds, dim=1).cpu().numpy()
            for i,pid in enumerate(patient_ids):
                patient_preds.setdefault(pid, []).append(label_map[int(best[i])])
    # voto por mayoría (devuelve sólo el label ganador)
    return {pid: max(set(v), key=v.count) for pid,v in patient_preds.items()}

def summarize_ensemble_predictions(predictions):
    """
    Recibe:
      predictions: dict { patient_id: {
                                    'votes': [label1, ..., labelN],
                                    'frame_paths': [ruta1, ..., rutaN] }

    Devuelve:
      DataFrame con columnas:
        - patient_id
        - final_label: voto mayoritario (o "Biopsy" si >= 4 votos)
        - agreement: fracción de votos que obtuvo el label ganador
        - votes: lista completa de votos
        - frame_paths: lista de rutas a los frames usados
    """
    rows = []
    for pid, data in predictions.items():
        votes = data["votes"]
        paths = data["frame_paths"]
        cnt = Counter(votes)

        # Regla extra: si hay al menos 4 votos "Biopsy", forzar ese label
        if cnt["Biopsy"] >= 4:
            final_label = "Biopsy"
            count = cnt["Biopsy"]
        else:
            final_label, count = cnt.most_common(1)[0]

        agreement = count / len(votes)
        rows.append({
            "patient_id": pid,
            "final_label": final_label,
            "agreement": agreement,
            "votes": votes,
            "frame_paths": paths
        })

    df = pd.DataFrame(rows)
    return df.sort_values("agreement", ascending=False)

