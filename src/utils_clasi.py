import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from model_lightning_clasi import MyModelMulticlass
import pandas as pd
from addict import Dict
from collections import Counter

def load_model(ckpt_path, model_name="resnet"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
      predictions: dict { patient_id: [label1, label2, ..., labelN] }
    Devuelve:
      DataFrame con columnas:
        - patient_id
        - final_label: voto mayoritario
        - agreement: fracción de votos que obtuvo el label ganador
        - votes: lista completa de votos
    """
    rows = []
    for pid, votes in predictions.items():
        cnt = Counter(votes)
        final_label, count = cnt.most_common(1)[0]
        agreement = count / len(votes)
        rows.append({
            "patient_id": pid,
            "final_label": final_label,
            "agreement": agreement,
            "votes": votes
        })
    df = pd.DataFrame(rows)
    return df.sort_values("agreement", ascending=False)

