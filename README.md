# Transformer-alapú nyelvi modell és szózsákalapú lineáris osztályozó összehasonlítása kórházi zárójelentések automatikus BNO-kódolásában
### Comparing Transformer-Based Language Model and Bag-of-Words Based Linear Classifier in Automated ICD Coding of Hospital Discharge Summaries

---

A projekt célja a kórházi zárójelentések automatikus BNO-kódolása mesterséges intelligencia segítségével. A kutatás során egy TF-IDF alapú logisztikus regresszió, valamint egy ModernBERT alapmodell teljesítménye került összehasonlításra. Ezentúl egy interaktív demonstrációs alkalmazás is létrehozásra került, amely képes részletesen bemutatni a döntéshozatali folyamatot.

The project aims to automate the ICD coding of hospital discharge summaries using artificial intelligence. During the research, the performance of a TF-IDF-based logistic regression and a ModernBERT base model was compared. Furthermore, an interactive demonstration application was developed to present the decision-making process in detail.

---

### Repozitórium struktúra / Repository structure



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

