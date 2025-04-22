# AI Health Assistant

**Desenvolupament d’un Assistènt de Salut Intel·ligent mitjançant Intel·ligència Artificial**

Aquest projecte té com a objectiu dissenyar i implementar un assistent de salut intel·ligent, basat en models de llenguatge preentrenats i ajustats amb dades específiques del domini de la salut. El sistema generarà recomanacions en format textual, oferint un pla estructurat i personalitzat per optimitzar el rendiment físic i millorar la qualitat de vida, incloent persones amb limitacions fisiques.

---

## Objectius del Projecte

1. Predir l'estat fisiològic de la persona (tensió / relaxació) per prevenir la fatiga i lesions.
2. Oferir recomanacions personalitzades a partir de les dades recollides, el context de la persona i l'estat predit.
3. Implementar un LLM ajustat al domini de la salut (Fine-tuning de GPT-4).
4. Proporcionar explicacions comprensibles i un pla d'acció diari.

---

## Arquitectura del Sistema

El projecte es divideix en dues etapes fonamentals:

### Etapa 1: Predicció de l'Estat Fisiològic

- Dataset: LifeSnaps
- Entrenament d’un model de *machine learning*.
- Predicció de l’estat de **tensió** i **relaxació** d’una persona.
- Utilització de dades fisiològiques (ex: HRV, ritme cardíac, patrons de son).

### Etapa 2: Fine-tuning del LLM

- Ajust de models de llenguatge amb dades del domini.
- Integració amb la predicció del model de ML.
- Generació de recomanacions i plans personalitzats.

---

## Tecnologies Utilitzades

- Python   
- Model LLM: GPT-4, DeepSeek-V3  
- APIs a LLMs 
- Pandas, Scikit-learn, Matplotlib (per EDA i ML)

---

## 📁 Estructura del Repositori

```
📦 AI-Health-Assistant/
├── data/                  # Dades utilitzades per entrenar / cleaned data
├── models/                # Model de ML i LLM
├── notebooks/             # Notebooks de proves: EDA, preprocessament, entrenament de models, etc.
├── src/                   # Codi font del projecte
│   ├── 01_preprocessing/  # Tractament de dades
│   ├── 02_training/       # Entrenament del model ML
│   └── 03_assistant/      # Mòdul del LLM i generació de respostes
├── results/               # Resultats, mètriques i figures finals
└── README.md              
```

---

## Estat del Desenvolupament

- [x] Definició de l'arquitectura
- [x] Preparació de les dades inicials
- [ ] Entrenament del model de predicció fisiològica
- [ ] Fine-tuning del LLM
- [ ] Integració i proves finals

---

