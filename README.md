# Human Activity Recognition - HAR Classification Project 📊

Este proyecto aplica técnicas de aprendizaje automático para clasificar seis actividades humanas (caminar, subir escaleras, bajar escaleras, estar sentado, de pie y acostado) usando señales del acelerómetro y giroscopio de un smartphone.

---

## 📁 Estructura del proyecto

classification-project/
│
├── data/ # Contiene train.h5 y test.h5 (ignorado por git)
├── models/ # Modelos entrenados (.joblib)
├── results/
│ └── figures/ # Gráficos generados automáticamente
├── src/ # Módulos de preprocesamiento y entrenamiento
│ ├── preprocessing_flattened.py
│ ├── preprocessing_tsfresh.py
│ ├── classification.py
│ ├── generate_submission.py
│ └── utils.py
├── main.py # Punto de entrada del proyecto
├── requirements.txt
└── README.md

---

## ⚙️ Requisitos

- Python ≥ 3.8
- Ver dependencias en `requirements.txt`

---

## 🚀 Cómo ejecutar

```bash
python main.py