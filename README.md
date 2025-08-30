# Nutmeg Fungal Disease Early Warning (IoT + ML + CV + GenAI-ready)

An end-to-end,ready project that simulates IoT sensor data for nutmeg farms, trains:
- a **sensor-based risk model** (Logistic Regression / Random Forest), and
- an **image classifier** (ResNet18 transfer learning),
then combines both to produce a final risk score and farmer-friendly recommendations. Includes a **Streamlit app**.

> Designed to map closely to your patent idea: *IoT-Based Identification of Fungal Disease in Nutmeg Leaves Using a Machine Learning Approach*.

## Quick Start

### 1) Create environment & install requirements
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Generate synthetic sensor data
```bash
python src/data/simulate_sensors.py --out data/processed/sensors.csv --days 30
```

### 3) Train sensor model
```bash
python src/models/train_sensor_model.py --data data/processed/sensors.csv --out models/sensor_model.pkl
```

### 4) Prepare image dataset (PlantVillage-like structure)
Create a folder like:
```
data/images/
    healthy/
        img1.jpg
        ...
    diseased/
        imgA.jpg
        ...
```

### 5) Train image classifier
```bash
python src/models/train_cnn.py --data_dir data/images --out models/cv_model.pt --epochs 3
```
This will also save `models/class_to_idx.json`.

### 6) Run Streamlit app
```bash
streamlit run src/app/ui_streamlit.py
```
Upload a leaf image and fill sensor values (or paste a row from `sensors.csv`). Get a risk score + recommendations.

---

## Project Structure
```
nutmeg-fungal-risk-project/
├── README.md
├── requirements.txt
├── src/
│   ├── app/
│   │   └── ui_streamlit.py
│   ├── data/
│   │   └── simulate_sensors.py
│   ├── models/
│   │   ├── train_sensor_model.py
│   │   ├── train_cnn.py
│   │   └── infer.py
│   └── utils/
│       ├── config.py
│       ├── data_utils.py
│       └── model_utils.py
├── data/
│   ├── raw/            # (optional) put raw dumps here
│   └── processed/      # sensors.csv is generated here
├── models/
│   ├── sensor_model.pkl
│   ├── cv_model.pt
│   └── class_to_idx.json
├── Dockerfile
└── .github/workflows/ci.yml
```
## License
MIT
