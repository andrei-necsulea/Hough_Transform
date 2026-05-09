# Hough Transform for Line Detection

Acest proiect implementeaza si compara mai multe variante ale Transformatei Hough pentru detectia liniilor in imagini alb-negru. Scopul este analiza diferentei dintre o varianta secventiala, o varianta paralela cu Numba/OpenMP si o varianta hibrida MPI + OpenMP, cu accent pe timpul de executie, speedup, eficienta si costul de sincronizare al acumulatorului Hough.

## Obiectivul proiectului

Proiectul este gandit pentru cerinta:

**OpenMP vs MPI + OpenMP pentru Transformata Hough**

Se urmareste:

- compararea unei implementari secventiale cu implementari paralele;
- analiza scalabilitatii in functie de numarul de thread-uri si procese;
- observarea impactului rezolutiei acumulatorului Hough asupra performantei;
- interpretarea rezultatelor pe baza unor metrici clare, nu doar a timpilor de executie.

## Descriere scurta a algoritmului

Transformata Hough pentru linii detecteaza drepte in imagine prin vot in spatiul parametrilor $(\rho, \theta)$.

Fluxul principal este:

1. Se incarca imaginea in tonuri de gri.
2. Se extrag muchiile cu Canny.
3. Pentru fiecare pixel de muchie, se calculeaza perechi $(\rho, \theta)$ pentru mai multe unghiuri.
4. Se incrementeaza acumulatorul Hough pentru fiecare combinatie validata.
5. Varfurile din acumulator indica liniile candidate.

Rolul acestui algoritm in procesarea imaginilor este de a transforma problema de detectie a liniilor din spatiul imaginii intr-o problema de vot in spatiul parametrilor, facand detectia mai robusta la zgomot si discontinuitati.

## Implementari disponibile

### 1. Varianta secventiala

Fisier: [src/hough_sequential/hough_core.py](src/hough_sequential/hough_core.py)

Aceasta varianta calculeaza acumulatorul Hough intr-o singura bucla, fara paralelizare. Este baza de comparatie pentru toate celelalte implementari.

### 2. Varianta paralela cu Numba

Fisier: [src/hough_numba/hough_numba.py](src/hough_numba/hough_numba.py)

Aceasta implementare foloseste `@njit(parallel=True)` si `prange` pentru a distribui procesarea pixelilor de muchie intre mai multe thread-uri. In spate, Numba poate folosi un model de tip OpenMP, in functie de mediu si configuratie.

Observatie importanta: acumulatorul Hough este o structura partajata, deci exista risc de race condition daca mai multe thread-uri scriu simultan in aceeasi celula. Acesta este un punct relevant de discutat in raportul final, deoarece sincronizarea poate afecta scalabilitatea.

### 3. Varianta hibrida MPI + OpenMP

Fisier: [src/hough_mpi.py](src/hough_mpi.py)

Aceasta varianta imparte imaginea intre procese MPI, iar fiecare proces calculeaza local acumulatorul folosind paralelizare pe thread-uri prin Numba/OpenMP. Rezultatele locale sunt combinate prin `MPI.Reduce`.

Acest model este util pentru analiza diferentei dintre:

- paralelizarea pe un singur nod, cu thread-uri multiple;
- paralelizarea pe mai multe procese, eventual pe mai multe noduri, cu reducerea costului de sincronizare prin agregare finala.

## UI si rulare interactiva

Fisier: [src/gui_tkinter.py](src/gui_tkinter.py)

Aplicatia GUI permite:

- alegerea unei imagini;
- selectarea implementarii (`Sequential` sau `Numba Parallel`);
- modificarea parametrilor `rho`, `theta` si pragului de linii;
- salvarea rezultatului in folderul `results/`;
- compararea rapida a rezultatelor si a timpilor de executie.

## Benchmark si metrici

Fisier: [src/benchmark/benchmark_seq_numba.py](src/benchmark/benchmark_seq_numba.py)

Benchmark-ul existent compara varianta secventiala cu varianta Numba si salveaza rezultatele in CSV.

Metricile urmarite in proiect sunt:

- timp de executie;
- speedup: $S = T_{secvential} / T_{paralel}$;
- eficienta: $E = S / P$, unde $P$ este numarul de thread-uri sau procese;
- cost de sincronizare al acumulatorului;
- impactul rezolutiei acumulatorului Hough asupra timpului si memoriei.

Pentru analiza finala, este recomandat sa se testeze mai multe configuratii:

- numar diferit de thread-uri;
- numar diferit de procese MPI;
- valori diferite pentru `rho_res` si `theta_res`;
- imagini de dimensiuni si complexitati diferite.

## Dataset

Dataset-ul poate fi ales de echipa, de preferat din Kaggle.

In prezent, proiectul foloseste o imagine de test din:

- [data/images/h1.png](data/images/h1.png)

Pentru raportul final, este bine sa mentionezi:

- sursa dataset-ului;
- numarul de imagini utilizate;
- dimensiunea acestora;
- de ce dataset-ul este relevant pentru detectia liniilor sau cercurilor.

## Structura proiectului

```text
README.md
requirements.txt
data/
	images/
src/
	benchmark/
		benchmark_seq_numba.py
	hough_mpi.py
	hough_numba/
		hough_numba.py
	hough_sequential/
		hough_core.py
	gui_tkinter.py
	image_utils.py
results/
```

## Cerinte de instalare

### Python

Se recomanda Python 3.10+.

### Dependinte

Instalare cu pip:

```bash
pip install -r requirements.txt
```

Pentru varianta MPI este necesar si un runtime MPI compatibil cu `mpi4py`.

## Cum rulezi proiectul

### 1. Varianta GUI

```bash
python src/gui_tkinter.py
```

### 2. Varianta secventiala / benchmark

Putem folosi functiile din pachetele din `src/` sau benchmark-ul dedicat:

```bash
python src/benchmark/benchmark_seq_numba.py
```

### 3. Varianta hibrida MPI + OpenMP

Executia trebuie facuta cu launcher-ul MPI al mediului:

```bash
mpiexec -n 4 python src/hough_mpi.py
```

Numarul de procese poate fi modificat pentru analiza scalabilitatii.

## Ce salveaza proiectul

- rezultate vizuale in folderul `results/`;
- CSV pentru benchmark-ul secvential vs Numba;
- timpi, speedup si eficienta afisate in consola sau in GUI.

## Configuratia hardware si software pentru raport

### Hardware

- CPU: AMD Ryzen 7, 7435HS
- Numar de nuclee / thread-uri: 8 cores / 16 threads
- RAM: 16GB DDR5, 4800MHZ
- GPU: NVIDIA RTX 4060, 8GB VRAM

### Software

- OS: Windows
- Python: 3.13.5
- NumPy
- OpenCV
- Matplotlib
- Numba
- mpi4py

## Ce trebuie discutat in raport

Raportul scris ar trebui sa includa:

- o descriere scurta a algoritmului si a rolului sau in procesarea imaginilor;
- strategiile de paralelizare implementate;
- comparatia intre implementarea secventiala si cele paralele;
- configuratia hardware si software utilizata;
- interpretarea personala a rezultatelor, nu doar tabele cu timpi.

Pentru aceasta tema, este important sa discuti si urmatoarele aspecte:

- de ce acumulatorul Hough devine un punct critic de sincronizare;
- cum se modifica performanta atunci cand creste rezolutia acumulatorului;
- daca speedup-ul creste liniar cu numarul de thread-uri / procese;
- unde apare overhead-ul de comunicare in cazul MPI;
- in ce scenariu varianta hibrida poate depasi varianta doar cu thread-uri.

## Branch-uri si livrabile finale

Proiectul este organizat cu branch-uri separate pentru variantele de implementare. Aceasta structura ajuta la urmarirea commit-urilor granulare si la prezentarea evolutiei proiectului.

Livrabilele finale cerute sunt:

- repository Git cu branch-uri separate pentru fiecare implementare;
- commit-uri descriptive si incrementale;
- raport scris(README.md)

## Observatii finale

Aceasta implementare este orientata pe detectia liniilor.
