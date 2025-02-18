# zadana heurystyka - pakujemy przedmioty według stosunku wartości do wagi

# liczymy (moze jakas funkcja z biblioteki) stosunek wartosci/waga z dwóch list

# stosunek wartości do wagi 
# znajdź największy -- zwroc jego id, sprawdz czy waga po dodaniu go nie przekracza plecaka 
# permutation 0000000 - wypełniona zerami do długości ilości przemiotów
# wkładanie do plecaka (czyli zaznaczanie w permutation 1 na elemencie indeksu) do momentu kiedy waga jest <= self.capacity
# jeżeli większa wartość to self.best_permutation = permutation, 
# returnowanie podobnie jak w poprzednim 

pw_ratios = [2.3, 1, 9.8, 0]
print(pw_ratios.index(max(pw_ratios)))