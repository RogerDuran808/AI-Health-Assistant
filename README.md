# AI Health Assistant

**Desenvolupament d’un Assistènt de Salut Intel·ligent mitjançant Intel·ligència Artificial**

Aquest projecte té com a objectiu dissenyar i implementar un assistent de salut intel·ligent, basat en models de llenguatge preentrenats i ajustats amb dades específiques del domini de la salut. El sistema generarà recomanacions en format textual, oferint un pla estructurat i personalitzat per optimitzar el rendiment físic i millorar la qualitat de vida, incloent persones amb limitacions fisiques.

---

## Objectius del Projecte

1. Predir l'estat fisiològic de la persona (tensió i cansament) per prevenir la fatiga i lesions.
2. Oferir recomanacions personalitzades a partir de les dades recollides, el context de la persona i l'estat predit.
3. Implementar un LLM ajustat al domini de la salut (Fine-tuning de GPT-4).
4. Proporcionar explicacions comprensibles i un pla d'acció diari.

---

## Arquitectura del Sistema

El projecte es divideix en dues etapes fonamentals:

### Etapa 1: Predicció de l'Estat Fisiològic

- Dataset: LifeSnaps
- Entrenament d’un model de *machine learning*.
- Predicció de l’estat de **tensió/ansietat** i **cansament** d’una persona.
- Utilització de dades fisiològiques (ex: HRV, ritme cardíac, patrons de son).

### Etapa 2: Fine-tuning del LLM

- Ajust de models de llenguatge amb dades del domini.
- Integració amb la predicció del model de ML.
- Generació de recomanacions i plans personalitzats.

---

## Tecnologies Utilitzades

- Python   
- Pandas, Scikit-learn, Matplotlib (per EDA i ML)

---

## 📁 Estructura del Repositori

```
AI-Health-Assistant/
├── data/                               # Dades utilitzades per entrenar / cleaned data
├── models/                             # Model de ML
├── notebooks/                          # Notebooks de proves: EDA, preprocessament, entrenament de models, etc.
├── results/                            # Resultats, mètriques i figures finals
├── src/
│   └── ai_health_assistant/            # Paquet Python instal·lable
│       ├── preprocessing/              # Tractament de dades
│       │   ├── clean_data.py
│       │   └── preprocess_data.py
│       ├── training/                   # Entrenament del model ML
│       │   └── model_training.py
│       ├── assistant/
│       │   └── ai_health_prediction.py
│       └── utils/
│           └── clean_helpers.py
├── tests/                                # Proves
├── setup.py                              # Configuració del paquet
└── README.md

```

---

🔧 Instal·lació ràpida (entorn de desenvolupament)
```
# 1. Clonar el projecte i entrar‑hi
git clone https://github.com/RogerDuran808/AI-Health-Assistant.git
cd AI-Health-Assistant

# 2. Crear i activar un entorn virtual
python -m venv .venv

# Windows: 
.venv\Scripts\Activate.ps1

# 3. Instal·lar el paquet en mode editable
pip install -e .
```

Després d’aquest pas podràs importar el codi des de qualsevol script o notebook, per exemple:
```
from ai_health_assistant.utils.clean_helpers import clean_data
```

## Estat del Desenvolupament

- [x] Definició de l'arquitectura
- [x] Preparació de les dades inicials
- [ ] Entrenament del model de predicció fisiològica
- [ ] Fine-tuning del LLM
- [ ] Integració i proves finals

### Nomenclatura de versions

- v0.1.0 → Projecte / notebook en proves
    - v0.1.1 → Correcció d'algun error o petites modificacions
- v1.0.0 → Primera versión completa del notebook o projecte
    - v1.1.0 → Funcions noves del projecte o notebook

---

