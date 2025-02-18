from itertools import product
import numpy as np

# popracować z numpy


class KnapSack:
    def __init__(self, profits, weights, capacity):
        self.profits = profits
        self.weights = weights
        self.capacity = capacity
        self.max_profit = 0
        self.best_permutation = ()

    def calculate_weight(self, permutation):
        return [a * b for a, b in zip(self.weights, list(permutation))]

    def calculate_profit(self, permutation):
        return [a * b for a, b in zip(self.profits, list(permutation))]

    @staticmethod
    def calculate_indexes(permutation):
        return np.where(np.array(permutation) == 1)[0]

    def solve_knapsack_brute_force(self):
        permutations = product(range(2), repeat=len(self.profits))
        for permutation in permutations:
            if sum(self.calculate_weight(permutation)) <= self.capacity:
                if sum(self.calculate_profit(permutation)) >= self.max_profit:
                    self.max_profit = sum(self.calculate_profit(permutation))
                    self.best_permutation = permutation
                    # co jeżeli jest więcej tak samo dobrych permutacji???
        return (list(self.calculate_indexes(self.best_permutation)),
                self.max_profit,
                sum(self.calculate_weight(self.best_permutation)))

    def calculate_profit_weight_ratio(self):
        return [a / b for a, b in zip(self.profits, list(self.weights))]

    def solve_knapsack_pw_ratio(self):
        pw_ratios = self.calculate_profit_weight_ratio()
        result = [0 * i for i in range(len(self.profits))]
        current_weight = 0
        while len(pw_ratios) > 0:
            max_index = pw_ratios.index(max(pw_ratios))
            if current_weight + self.weights[max_index] <= self.capacity:
                result[max_index] = 1
                current_weight += self.weights[max_index]
            # zawsze usuwamy przetestowany z listy dopóki lista nie jest pusta
            pw_ratios.pop(max_index)
        return (list(self.calculate_indexes(result)),
                sum(self.calculate_profit(result)),
                current_weight)


weights = np.array([8, 3, 5, 2])
capacity = 9
profits = np.array([16, 8, 9, 6])

ks = KnapSack(profits, weights, capacity)
print(ks.solve_knapsack_brute_force())
print(ks.solve_knapsack_pw_ratio())

# Dla obu metod otrzymałam różne rozwiązania. Po obserwacji mogę stwierdzić, że bardziej dokładne wyniki
# zapewniła metoda brute_force. Brała ona pod uwagę wszystkie możliwości, z których następnie wybierała najbardzeiej optymalną.
# Metoda wyznaczenie współczynnika stosunku wartości do wagi okazała się mniej dokłądna. Brała ona pod uwagę jedynie te permutacje - układy,
# które zawierały kolejne elementy o największym współczynniku. W ten sposób z góry założyliśmy, że optymalne rozwiązanie będzie zawierać 
# element o najleoszym współczynniku, co w zależności od wartości pozostałych elementów może okazać się nieprawidłowe. 
# Ta heurystyka osobno ze zbioru odszukuje najbarzdziej optymlany element w danym momencie i zakłada, że zostanie on częścią rozwiązania, jednka nie bierze pod uwagę, 
# że ostatecznie najbardziej optymlne może być połaczenie kilku mniej "wartosciowo - wagowo" optymalnych elementów.

# stosunek wartości do wagi
# znajdź największy -- zwroc jego id, sprawdz czy waga po dodaniu go nie przekracza plecaka 
# permutation 0000000 - wypełniona zerami do długości ilości przemiotów
# wkładanie do plecaka (czyli zaznaczanie w permutation 1 na elemencie indeksu) do momentu kiedy waga jest <= self.capacity
# jeżeli większa wartość to self.best_permutation = permutation, 
# returnowanie podobnie jak w poprzednim 



# mamy podane dwie listy - wagi i wartosci przedmitow oraz maksymalna
# mase plecaka
# maksymalną wartość plecaka ustwaimay na 0

# najlepsza permutacja 0000 -> tyle razy ile wynosi size podanych list - ilość
# przedmiotó

# funkcją product z biblioteki intertools generujemyu wszystkie permutacje
# (musze obczaic jak ta funkcja działa, czy zwraca listę, co należy do niej
# podać)
# ZWRACA ITERATOR PO KTÓRYM BEDZIEMY CHODZIĆ

# przeszukujemy kazda permutacje (iterujemy po liscie wag i mnożemy razy numer
# indeksu w permutacji (czyli razy 1 lub 0))

# jezeli mniejsz rowna wadze przechodzimy dalej: podobne przeszukania dla
# wartości
# sprawdzenie czy wartosc sumerayczna jest wieksza od obecnego maksa, jesli
# tak zamiana maksa i zapamietanie permutacji

# zwracamy te indeksy na któey w najlepszej permutacji jest jedynka (w liście)
# policzenie funkcja do obliczania masy najlepszej permutacj
# zwracamy indeksy przedmiotów zapakowanych, wartość plecaka i mase plecaka

# funkcja do obliczania masy i do obliczania wartości
