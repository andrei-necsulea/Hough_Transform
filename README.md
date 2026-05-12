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

Benchmark-ul de baza compara varianta secventiala cu varianta Numba si salveaza rezultatele in CSV. In rularea extinsa a proiectului au fost comparate si implementarea MPI, respectiv varianta hibrida MPI + Numba, atat pentru o singura imagine, cat si pentru un folder cu 25 de imagini.

Metricile urmarite in proiect sunt:

- timp de executie;
- speedup: $S = T_{secvential} / T_{paralel}$;
- eficienta: $E = S / P$, unde $P$ este numarul de thread-uri sau procese;
- cost de sincronizare al acumulatorului;
- impactul rezolutiei acumulatorului Hough asupra timpului si memoriei.

Pentru interpretarea rezultatelor, este important de observat ca speedup-urile foarte mari ale Numba si ale variantei hibride includ si eliminarea overhead-ului Python prin JIT. Din acest motiv, cifrele trebuie citite impreuna cu timpul absolut, nu doar ca raport fata de varianta secventiala.

### Rezultate experimentale extinse

#### 1. Single image: h1.png

Pe imaginea de test individuala, cu 5331 pixeli de muchie si acumulator de forma (1401, 180), s-au obtinut urmatoarele rezultate:

- Sequential: 1.620921 s
- Numba Parallel: 0.000723 s
- MPI, 2 procese: 0.811546 s
- Hybrid MPI + Numba, 2 procese: 0.001916 s
- MPI, 4 procese: 0.431147 s
- Hybrid MPI + Numba, 4 procese: 0.001929 s

Concluzii pentru o singura imagine:

- Numba Parallel: 2242.25x speedup, eficienta 140.1405
- MPI, 2 procese: 2.00x speedup, eficienta 0.9987
- Hybrid MPI + Numba, 2 procese: 845.99x speedup, eficienta 422.9961
- MPI, 4 procese: 3.76x speedup, eficienta 0.9399
- Hybrid MPI + Numba, 4 procese: 840.29x speedup, eficienta 210.0727

Observatie practica: MPI scaleaza aproape liniar de la 2 la 4 procese pe aceasta masina, dar ramane mult mai lent decat Numba din cauza overhead-ului de comunicare, scatter si reduce. Varianta Numba si varianta hibrida par supra-liniare deoarece timpul secvential include overhead-ul Python, iar partea compilata JIT este masurata separat.

#### 2. Batch folder: 25 imagini

Pentru procesarea folderului de date, s-au obtinut urmatoarele valori agregate:

- Sequential: 193.118799 s total, 7.724752 s/ imagine
- MPI: 104.778244 s total, 4.191130 s/ imagine
- Hybrid MPI + Numba: 0.119396 s total, 0.004776 s/ imagine
- Numba Parallel: 0.080625 s total, 0.003225 s/ imagine

Speedup total fata de Sequential:

- MPI: 1.84x
- Hybrid MPI + Numba: 1617.46x
- Numba Parallel: 2395.27x

Interval observat pentru imaginile din batch:

- pixeli de muchie: 8601 - 54125 pe imagine;
- timpi MPI: 1.325016 s - 8.002854 s pe imagine;
- timpi Hybrid MPI + Numba: 0.001655 s - 0.007668 s pe imagine;
- timpi Numba Parallel: 0.001954 s - 0.004272 s pe imagine.

Pentru aceste imagini, acumulatorul Hough a ramas de forma (1159, 180) la rezolutia folosita in benchmark (`rho_res=1`, `theta_res=π/180`), ceea ce a mentinut comparatia consistenta intre implementari.

Pentru analiza finala, este recomandat sa se testeze mai multe configuratii:

- numar diferit de thread-uri;
- numar diferit de procese MPI;
- valori diferite pentru `rho_res` si `theta_res`;
- imagini de dimensiuni si complexitati diferite.

## Dataset

Proiectul utilizeaza 5 resurse de date:

### 1. Imagine de test inițială

- **Path**: [data/images/h1.png](data/images/h1.png)
- **Scop**: Imagine de test rapid pentru debugging și validare rapidă
- **Utilizare**: Benchmark single-image cu rezultate cunoscute

### 2. BSDS500 (Berkeley Segmentation Dataset)

- **Path**: [data/images/BSR/](data/images/BSR/)
- **Descriere**: Dataset standard pentru evaluarea algoritmilor de detecție de contur și segmentare
- **Relevanță**: Conține imaginea cu margini bine definite, ideale pentru testarea Transformatei Hough
- **Source**: Berkeley Vision and Learning Center

### 3. Road Lane Detection Dataset

- **Path**: [data/images/road_lane_detection_dataset/](data/images/road_lane_detection_dataset/)
- **Descriere**: Imagini de drumuri cu lini de marcaj (lane markers)
- **Relevanță**: Aplicație practică a Transformatei Hough - detecția liniilor în imagini de conducere autonomă
- **Tip de linii**: Linii paralele și linii de intersecție

### 4. Massachusetts Buildings Dataset

- **Path**: [datasets/massachusetts_buildings_dataset/](datasets/massachusetts_buildings_dataset/)
- **Descriere**: Imagini aeriene cu clădiri și contururi pe hartă
- **Relevanță**: Detecția marginilor clădirilor și a liniilor structurale în imagini de rezoluție înaltă
- **Scop**: Analiza performanței pe imagini cu structuri geometrice complexe

### 5. TuSimple Lane Detection Dataset

- **Path**: [datasets/tusimple_lane_detection_dataset/](datasets/tusimple_lane_detection_dataset/)
- **Descriere**: Dataset industri pentru detecția de linii în conducerea autonomă
- **Relevanță**: Imagini real-world cu condiții variate de iluminare și congestie
- **Sursa**: TuSimple (companie de tehnologie pentru conducere autonomă)

### Statistica dataset-ului utilizat în benchmark:

Din rezultatele obținute:

- **25 imagini** au fost procesate din dataset
- **Dimensiuni**: 321x481 sau 481x321 pixeli (imagini standard pentru procesare video)
- **Pixeli de muchie**: 8,601 - 54,125 pixeli per imagine
- **Acumulator**: Rezoluție constantă (1,159, 180) pentru imaginile din batch; pe h1.png, forma este (1,401, 180)

### Relevanță pentru Transformata Hough:

1. **Detecția de linii structurale**: Dataset-urile conțin linii clare (drumuri, margini de clădiri, contururi)
2. **Variabilitate**: Mix de imagini cu conținut diferit (drumuri, clădiri, contururi) permite evaluare robustă
3. **Aplicabilitate practică**: Scenarii reale din conducerea autonomă și analiza imagine
4. **Complexitate variabilă**: De la imagini simple (h1.png) la imagini complexe cu multi-linii (Massachusetts, TuSimple)

## Structura proiectului

```text
README.md
requirements.txt
data/
	images/
		h1.png
		BSR/
			bench/
				benchmarks/
				data/
					groundTruth/
					images/
					png/
					segs/
					test_1/, test_2/, test_3/, test_4/, test_5/
					ucm2/
				source/
		BSDS500/
			data/
				groundTruth/
				images/
		documentation/
		road_lane_detection_dataset/
datasets/
	massachusetts_buildings_dataset/
	tusimple_lane_detection_dataset/
results/
	batch_outputs/
	dataset_outputs/
	benchmark_seq_numba.csv
	dataset_benchmark_seq_numba.csv
	dataset_selected_implementation.csv
	benchmark_comparison.png
	dataset_comparison_chart.png
	single_image_benchmark_comparison.png
	*.jpg (imagini rezultate)
src/
	benchmark/
		benchmark_seq_numba.py
	hough_mpi.py
	hough_numba/
		hough_numba.py
		run_numba.py
	hough_sequential/
		hough_core.py
		hough_sequential.py
	dataset_manager.py
	gui_tkinter.py
	image_utils.py
```

### Descriere directoare:

- **data/images/**: Imagini de test și dataset-uri locale (h1.png, BSR, BSDS500, road_lane_detection_dataset)
- **datasets/**: Dataset-uri externe mari (Massachusetts Buildings, TuSimple Lane Detection)
- **src/**: Codul sursă cu implementări (sequential, numba, mpi) și utilități
- **results/**: Rezultate benchmark (CSV), grafice și imagini de output cu liniile detectate
- **results/batch_outputs/**: Output-uri din procesare batch
- **results/dataset_outputs/**: Imagini cu liniile detectate din fiecare imagine din dataset

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

# Observatii

### 1. Acumulatorul Hough ca punct critic de sincronizare

Transformata Hough se bazează pe o structură comună de date (acumulatorul), unde fiecare pixel de muchie contribuie la mai multe celule. În varianta secvențială, această scriere este simplă. În varianta paralelă, diferența dintre timpi arată că overhead-ul dominant nu este doar actualizarea acumulatorului, ci și costul de execuție Python și, pentru MPI, costul de comunicare.

Pe datele măsurate aici:

- Numba Parallel ajunge la 2242.25x speedup pe o singură imagine;
- Hybrid MPI + Numba rămâne foarte rapid, dar sub Numba pur, cu 845.99x la 2 procese și 840.29x la 4 procese;
- MPI singur este aproape liniar între 2 și 4 procese, dar rămâne departe de performanța thread-urilor compilate, cu doar 2.00x și 3.76x speedup pe h1.png.

### 2. Impactul rezoluției acumulatorului Hough asupra performanței

În experiment, rezoluția acumulatorului a fost fixată la o formă constantă de **(1159, 180)** pentru toate imaginile din dataset, corespunzând unui `rho_res=1` și `theta_res=π/180` (1 grad).

Observații cheie:

- Pentru imagini cu **mai mulți pixeli de muchie** (54,125 pixeli în cazul 108004.jpg), timpul crește, dar imaginea rămâne în același ordin de mărime pentru implementările paralele; de exemplu, pe batch, MPI variază între 1.325016 s și 8.002854 s, iar Numba între 0.001954 s și 0.004272 s.
- Rezoluția mai fină ar duce la un acumulator mai mare și, probabil, la mai mult timp de acces la memorie, dar acest compromis nu a fost măsurat aici.
- Rezoluția mai grosieră ar putea crește **false positives** (suprapunere de linii în spatiul parametrilor).

### 3. Scalabilitatea speedup-ului cu numărul de thread-uri

CPU-ul utilizat are **16 thread-uri logice (8 nuclee fizice)**. Speedup-ul observat este **2242.25x pe o singură imagine** și **1617.46x - 2395.27x pe batch**, în funcție de implementare. Aceste valori depășesc mult 16x, deci ele nu reflectă doar paralelizare brută, ci și eliminarea overhead-ului Python prin Numba și efectul de compilare JIT.

Cauze posibile ale acestui speedup extraordinar:

1. **Efectul JIT (Just-In-Time Compilation) de Numba**: Compilarea la nivel de mașină elimină overhead-ul interpretării Python și permite optimizări agresive.
2. **Cache Locality**: Organizarea iterației prin pixeli (cu `prange` pe y, apoi x pe lățimea cu cache-ul) reduce miss-urile de cache.
3. **Vectorizare automată**: Numba poate utiliza instrucțiuni SIMD (AVX-2) pe CPU-ul Ryzen.
4. **Reducerea overhead-ului**: Varianta Numba nu are overhead de inițializare Python/NumPy în interior.

### 4. Overhead-ul de comunicare în cazul MPI

Implementarea MPI+OpenMP din cod include:

- **Broadcast**: Dimensiunile imaginii sunt trimise tuturor proceselor.
- **Scatter**: Pixelii de muchie sunt distribuiți în mod egal între procese.
- **Reduce**: Acumulatoarele locale sunt însumate la procesul root.

În scenariile cu procese pe un singur nod, costul de comunicare este vizibil mai mare decât la thread-uri. Acest lucru se vede direct în cifre: pe h1.png, MPI cu 2 procese are 0.811546 s, iar cu 4 procese 0.431147 s, în timp ce Numba rămâne la 0.000723 s. Pe batch, MPI totalizează 104.778244 s, față de 0.119396 s pentru varianta hibridă și 0.080625 s pentru Numba. Dacă procesele ar fi distribuite pe mai multe noduri, latența de rețea ar deveni și mai relevantă.

### 5. Scenario-uri în care varianta hibridă (MPI + OpenMP) poate depăși varianta doar cu thread-uri

Varianta hibridă este benefică în situații precum:

- **Calcul pe cluster-e distribuite**: Dacă imaginile sunt prea mari sau numărul de pixeli de muchie este extrem de mare, distribuția pe mai multe noduri cu MPI permite paralelizare la scară mai mare.
- **Echilibrare de sarcină**: MPI poate distribui munca mai bine dacă fiecare proces are mașini heterogene.
- **Evitarea congestiei memoria**: O singură mașină cu 16 thread-uri poate fi limitată de lățimea de bandă a memoriei. MPI pe mai multe noduri oferă lățime de bandă distribuită.

Totuși, pentru imaginile testate aici, varianta Numba cu thread-uri locale este cea mai rapidă pe aceeași mașină, iar hibridul rămâne mai degrabă util ca punct de extensie pentru cluster-e, nu ca soluție optimă pe un singur nod.

## Branch-uri si livrabile finale

Proiectul este organizat cu branch-uri separate pentru variantele de implementare. Aceasta structura ajuta la urmarirea commit-urilor granulare si la prezentarea evolutiei proiectului.

Livrabilele finale cerute sunt:

- repository Git cu branch-uri separate pentru fiecare implementare;
- commit-uri descriptive si incrementale;
- raport scris(README.md)

## Observatii finale

### Rezultate și concluzii principale

Această implementare a demonstrat succesul paralelizării Transformatei Hough prin compararea sistematică a patru variante relevante: secvențială, Numba/OpenMP, MPI și MPI+OpenMP.

#### Performanța măsurată:

**Benchmark single-image (h1.png - 5,331 pixeli de muchie):**

- Sequential: 1.620921 secunde
- Numba Parallel: 0.000723 secunde
- MPI, 2 procese: 0.811546 secunde
- Hybrid MPI + Numba, 2 procese: 0.001916 secunde
- MPI, 4 procese: 0.431147 secunde
- Hybrid MPI + Numba, 4 procese: 0.001929 secunde

**Batch folder (25 imagini cu 8,601-54,125 pixeli de muchie fiecare):**

- Sequential: 193.118799 secunde total, 7.724752 secunde/imagine
- Numba Parallel: 0.080625 secunde total, 0.003225 secunde/imagine
- MPI: 104.778244 secunde total, 4.191130 secunde/imagine
- Hybrid MPI + Numba: 0.119396 secunde total, 0.004776 secunde/imagine
- Speedup total Numba: 2395.27x
- Speedup total Hybrid: 1617.46x
- Speedup total MPI: 1.84x

#### Insight-uri importante:

1. **Compilarea JIT elimină overhead-ul Python**: diferența dintre 1.620921 s și 0.000723 s pe h1.png arată că o mare parte din costul secvențial vine din execuția Python, nu doar din algoritm.
2. **MPI pe un singur nod are overhead vizibil**: rezultatul de 0.811546 s la 2 procese și 0.431147 s la 4 procese confirmă că distribuirea muncii ajută, dar nu poate concura cu thread-urile compilate când datele sunt mici.
3. **Scalabilitatea este clară, dar diferită pe fiecare implementare**: MPI trece de la 2.00x la 3.76x când dublezi numărul de procese, în timp ce Numba și hybrid rămân în zona de timp foarte mic, dominată de compilare, cache și overhead minim în secțiunea măsurată.
4. **Batch-ul confirmă diferența dintre MPI și thread-uri compilate**: pe 25 de imagini, Numba este de 2395.27x mai rapid decât secvențial, hybrid-ul de 1617.46x, iar MPI de doar 1.84x.

#### Limitări și direcții viitoare:

1. **MPI nu a fost testat pe cluster real**: implementarea MPI este prezentă, dar testele practice au fost doar pe o singură mașină. Pe un cluster real, comunicarea ar introduce latență semnificativă.
2. **Rezoluția acumulatorului este fixă**: un studiu complet ar include variații ale `rho_res` și `theta_res` pentru a analiza trade-off-ul între acuratețe și performanță.
3. **Puterea GPU nu a fost exploatată**: NVIDIA RTX 4060 din sistem ar putea accelera și mai mult Hough Transform-ul prin CUDA (Numba suportă `target='cuda'`).
4. **Thread pinning și NUMA**: pe sisteme NUMA (Non-Uniform Memory Access), pinning-ul thread-urilor la anumite nuclee ar putea îmbunătăți performanța ulterior.

#### Aplicabilitate practică:

Paralelizarea Transformatei Hough cu Numba/OpenMP este **extrem de eficace** pentru aplicații real-time pe aceeași mașină:

- detecția de linii este practic instantanee în varianta compilată;
- scalabilitatea pe 25 de imagini rămâne foarte bună;
- codul este ușor de integrat în pipeline-uri Python existente.

MPI+OpenMP rămâne relevant pentru scenariile de calcul pe cluster-e, în special pentru imagini de mari dimensiuni sau rezoluții înalte ale acumulatorului, unde costul de comunicare poate fi amortizat de volum.

### Concluzii finale

1. **Numba cu `@njit(parallel=True)` oferă cea mai bună performanță pe aceeași mașină, cu 2242.25x pe o singură imagine și 2395.27x pe batch.**
2. **MPI scalează corect între 2 și 4 procese, dar pe acest workload rămâne mult în urma variantelor compilate cu thread-uri.**
3. **Varianta hibridă este utilă arhitectural, dar pe o singură mașină nu depășește Numba pur.**
4. **Pentru această implementare și acest set de date, factorul decisiv nu este doar numărul de procese/thread-uri, ci și costul efectiv al execuției Python versus cod compilat.**
