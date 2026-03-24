# Projekt 1 — Klasyfikacja i regresja dla danych tabelarycznych

Projekt realizowany w ramach przedmiotu Wprowadzenie do Sztucznej Inteligencji.

## Wymagania

- Python 3.13+

Instalacja zależności:

```bash
pip install -r requirements.txt
```

## Struktura projektu

```
WDSI-P1/
├── common/
│   └── DataManager.py              # Bazowa klasa do zarządzania danymi
├── classification/
│   ├── data/
│   │   ├── ortodoncja.csv           # Zbiór danych klasyfikacyjnych
│   │   └── ortodoncja dokumentacja.txt
│   ├── DataOrtho.py                 # Klasa danych (dziedziczy po DataManager)
│   ├── OrthoModel.py                # Modele klasyfikacyjne
│   └── DataOrthoAnalyser.py         # Główny skrypt klasyfikacji
└── regression/
    ├── data/
    │   ├── domy.csv                 # Zbiór danych regresyjnych
    │   └── domy dokumentacja.txt
    ├── DataHouses.py                # Klasa danych (dziedziczy po DataManager)
    ├── DataHousesConfig.py          # Konfiguracja modeli i hiperparametrów
    ├── DataHousesEvaluation.py      # Wyszukiwanie hiperparametrów i ewaluacja
    ├── DataHousesPlots.py           # Generowanie wykresów
    ├── DataHousesReport.py          # Generowanie raportu tekstowego
    ├── DataHousesAnalyser.py        # Główny skrypt regresji
    └── results/
        ├── attempt1/                # Wyniki próby 1
        ├── attempt2/                # Wyniki próby 2 
        └── attempt3/                # Wyniki próby 3 

```

---

## Klasyfikacja

### Uruchomienie

```bash
python -m classification.DataOrthoAnalyser
```

Skrypt wczytuje zbiór `ortodoncja.csv`, przeprowadza analizę danych, preprocessing, trenuje modele klasyfikacyjne i zapisuje wyniki do folderu `classification/analysis/`.

---

## Regresja

### Uruchomienie

```bash
python -m regression.DataHousesAnalyser
```

Wyniki zapisywane są do folderu wskazanego w `DataHousesConfig.py` (zmienna `RESULTS_DIR`).

### Odtwarzanie wyników poszczególnych prób

Każdy folder `regression/results/attemptX/` zawiera kopie plików `DataHouses.py` i `DataHousesConfig.py` użytych w danej próbie. Aby odtworzyć wyniki konkretnej próby należy:

1. Skopiować `regression/results/attemptX/DataHouses.py` do `regression/DataHouses.py`
2. Skopiować `regression/results/attemptX/DataHousesConfig.py` do `regression/DataHousesConfig.py`
3. Uruchomić `python -m regression.DataHousesAnalyser`

W docelowym archiwum ZIP foldery `attemptX` nie zawierają wygenerowanych wykresów ani plików CSV z podziałem danych — zostaną one wygenerowane automatycznie przy uruchomieniu programu. Przechowywane są jedynie pliki konfiguracyjne (`DataHouses.py`, `DataHousesConfig.py`) oraz raport tekstowy z wynikami (`regression_results.txt`).