# Deep Learning for population structure prediction

Projekt został zrealizowany jako projekt semestralny na przedmiot Metody Bioinformatyki na Wydziale Elektroniki i Technik Informacyjnych Politechniki Warszawskiej


## 1. Cel projektu

Celem projektu jest implementacja algorytmu opartego na głębokim uczeniu, który będzie
przewidywał (lub grupował) docelowe populacje z wysoką dokładnością przewidywalności i porówna
uzyskane rezultaty z istniejącymi.

## 2. Wymagane narzędzia do uruchomienia programu

* Python 3.5
* IDE np. PyCharm Community 2017
* Pandas
* NumPy
* scikit-learn
* seaborn
* scikit-allel

## 3. Uruchomienie programu

1. Otwieramy plik nn.py i definiujemy parametry sieci, które chcemy przetestować np.
batch_size = 128
max_nb_of_iterations = 200
learning_rate = 0.001

2. Definiujemy ilość wybranych cech do przetestowania naszego zbioru danych 
Zmienna data odpowiada za ilość wczytywanych do sieci cech np. 
data = np.array(dataset.iloc[:, 3:200]) oznacza wczytanie danych z 200 pierwszymi cechami do naszej sieci

3. Uruchamiamy plik main.py z dowolną liczbą parametrów typu całkowitoliczbowych oddzielonych spacjami, oznaczających ilość neurnów kolejnych warstw ukrytych

4. Czekamy na rezultat


