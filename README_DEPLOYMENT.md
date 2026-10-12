# Network Security Attack Detection — Streamlit

## Files you must add
After running `preprocessing.ipynb` and the helper cell, place these files in `model/`:
- `scaler.pkl`
- `feature_columns.pkl`
- `label_encoder.pkl`

After running `model_training.ipynb`, copy `security_attacks_model.keras` into `model/`.

Expected project structure:
```
network-security-detection/
├── app.py
├── requirements.txt
└── model/
    ├── security_attacks_model.keras
    ├── scaler.pkl
    ├── feature_columns.pkl
    └── label_encoder.pkl
```

## Run locally (Windows)
Open Command Prompt in this folder and run:
```
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Upload an NSL-KDD CSV/TXT file in the original 43-column format (41 traffic features, attack label, and level). The app ignores the label and difficulty-level columns for prediction. It expects the same one-hot feature layout used in training.

## Deploy
Push `app.py`, `requirements.txt`, and the `model/` artifacts to a GitHub repository. Then go to https://share.streamlit.io/, select **Create app**, choose the repository and branch, set the main file to `app.py`, and deploy.

Note: do not commit the original large dataset unless you need it for another purpose. The model and preprocessing artifacts are enough for inference.
