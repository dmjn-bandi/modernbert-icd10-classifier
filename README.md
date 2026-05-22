# Transformer-alapú nyelvi modell és szózsákalapú lineáris osztályozó összehasonlítása kórházi zárójelentések automatikus BNO-kódolásában

### Comparing Transformer-Based Language Model and Bag-of-Words Based Linear Classifier in Automated ICD Coding of Hospital Discharge Summaries

---

A projekt célja a kórházi zárójelentések automatikus BNO-kódolása mesterséges intelligencia segítségével. A kutatás során egy TF-IDF alapú logisztikus regressziós modell, valamint a ModernBERT alapmodelll teljesítménye került összehasonlításra. Ezentúl egy interaktív demonstrációs alkalmazás is létrehozásra került, amely képes részletesen bemutatni a döntéshozatali folyamatot.

The project aims to automate the ICD coding of hospital discharge summaries using artificial intelligence. During the research, the performance of a TF-IDF-based logistic regression model and the ModernBERT base model was compared. Furthermore, an interactive demonstration application was developed to present the decision-making process in detail.

---

### Repozitórium struktúra / Repository structure

```
├── app/
│   └── app.py
├── notebooks/
│   ├── analysis_hu.ipynb
│   ├── baseline_train.ipynb
│   ├── modernbert_fine_tune.ipynb
│   └── preprocess.ipynb
├── src/
│   ├── utils.py
│   └── __init__.py
├── .env.example
├── .gitignore
├── modernbert_download.py
├── README.md
├── requirements.txt
└── t_s_config.json
```

---

### Telepítés és használat / Setup and Usage

1. Függőségek telepítése / Install dependencies
   ```
   pip install -r requirements.txt
   ```
2. ModernBERT alapmodell letöltése / Download ModernBERT base model
   ```
   python modernbert_download.py
   ```
3. Környezeti változók beállítása / Set environment variables

   Hozz létre egy .env fájlt (env.example), és add meg benne a táblákhoz szükséges elérési útvonalakat. /
   Create a .env file (env.example) and provide the necessary file paths for the data tables.

4. Konfiguráció (Opcionális) / Configuration (Optional)
   
   A t_s_config.json fájlban igény szerint módosíthatod a beállításokat, például meghatározhatod, hogy pontosan mely zárójelentés szekciók kerüljenek be az egyes adathalmazokba. /
   You can optionally modify the t_s_config.json file to specify exactly which discharge summary sections should be included in the datasets.

5. Adatelőfeldolgozás / Data Preprocessing

   Futtasd le a ```notebooks/preprocess.ipynb``` notebookot. / Run the ```notebooks/preprocess.ipynb``` notebook.

6. Modellek betanítása / Model Training

   Futtasd le a ```notebooks/baseline_train.ipynb``` és ```notebooks/modernbert_fine_tune.ipynb``` notebookokat. / Run the ```notebooks/baseline_train.ipynb``` and ```notebooks/modernbert_fine_tune.ipynb``` notebooks.

7. Demonstrációs alkalmazás futtatása / Running the demo application

   ```
   python app/app.py
   ```


---

### Hivatkozások / Citations

```bibtex
@article{PhysioNet-mimiciv-3.1,
  author = {Johnson, Alistair and Bulgarelli, Lucas and Pollard, Tom and Gow, Brian and Moody, Benjamin and Horng, Steven and Celi, Leo Anthony and Mark, Roger},
  title = {{MIMIC-IV}},
  journal = {{PhysioNet}},
  year = {2024},
  month = oct,
  note = {Version 3.1},
  doi = {10.13026/kpb9-mt58},
  url = {https://doi.org/10.13026/kpb9-mt58}
}

@article{PhysioNet-mimic-iv-note-2.2,
  author = {Johnson, Alistair and Pollard, Tom and Horng, Steven and Celi, Leo Anthony and Mark, Roger},
  title = {{MIMIC-IV-Note: Deidentified free-text clinical notes}},
  journal = {{PhysioNet}},
  year = {2023},
  month = jan,
  note = {Version 2.2},
  doi = {10.13026/1n74-ne17},
  url = {https://doi.org/10.13026/1n74-ne17}
}

@misc{modernbert,
      title={Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference}, 
      author={Benjamin Warner and Antoine Chaffin and Benjamin Clavié and Orion Weller and Oskar Hallström and Said Taghadouini and Alexis Gallagher and Raja Biswas and Faisal Ladhak and Tom Aarsen and Nathan Cooper and Griffin Adams and Jeremy Howard and Iacopo Poli},
      year={2024},
      eprint={2412.13663},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2412.13663}, 
}

