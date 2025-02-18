import numpy as np
import matplotlib.pyplot as plt
import random


def min_max_norm(val, min_val, max_val, new_min, new_max):
    return (val - min_val) * (new_max - new_min) / (max_val - min_val) + new_min


class Chromosome:
    def __init__(self, length, array=None):
        # if array is None it should be initialized with random binary vector
        if array is None:
            array = np.random.randint(2, size=length)
        self.array = array
        self.arguments = None
        self.fitness = None

    def decode(self, lower_bound, upper_bound, aoi):
        array_part = self.array[lower_bound:upper_bound + 1]
        # decimal = np.packbits(array_part) -----------> zawsze konwertuje na liczbę dopełniając do 8 bitów
        decimal = 0
        for bit in array_part:
            decimal = decimal * 2 + bit
        return min_max_norm(decimal,  0, 2 ** len(array_part) - 1, aoi[0], aoi[1])

    # czemu ona tam czase zamienia mi istniejący obiekt na none, jeśli nie ma mutacji
    def mutation(self, probability):
        if np.random.rand() < probability:
            # Jeśli losowo wybrane prawdopodobieństwo jest mniejsze niż zadane
            index_to_mutate = np.random.randint(len(self.array))  # Losowy indeks genu do mutacji
            self.array[index_to_mutate] = 1 - self.array[index_to_mutate]

    def crossover(self, other):
        # point = random.randint(0, min(len(self.array), len(other.array)))
        # chyba tak też git, bo one i tak muszą być tej samej długości
        point = random.randint(0, len(self.array))
        # nie wiem czy indeks 0 ma sens, bo wtedy się naprawdę nie krzyżują
        # losowy punkt od 0 do długości krótszego z ciągów
        descendant_array = np.concatenate([self.array[:point], other.array[point:]])
        # descendant_2_array = np.concatenate(other.array[:point], self.array[point:])
        descendant_1 = Chromosome(len(self.array), descendant_array)
        # descendant_2 = Chromosome(len(self.array), descendant_2_array)
        return descendant_1
        # return descendant_1, descendant_2
        #  nie wiem czy tu zwracać jednego (obojętnie którego potomka), czy obu !!!!!!!!!!!!!!!
        #  CZY ZWRACAĆ LEPSZEGO?????


class GeneticAlgorithm:
    def __init__(self, chromosome_length, obj_func_num_args, objective_function, aoi, population_size=1000,
                 tournament_size=2, mutation_probability=0.05, crossover_probability=0.8, num_steps=30):
        assert chromosome_length % obj_func_num_args == 0, "Number of bits for each argument should be equal"
        self.chromosome_lengths = chromosome_length
        self.obj_func_num_args = obj_func_num_args
        self.bits_per_arg = int(chromosome_length / obj_func_num_args)
        self.objective_function = objective_function
        self.aoi = aoi
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.mutation_probability = mutation_probability
        self.crossover_probability = crossover_probability
        self.num_steps = num_steps

        self.population = self.get_population()
        self.best_of_generation = []
        # wszystkie informacje (obiekt, trace, wartość funkcji celu) o najlepszych

    def get_population(self):
        population = []
        for individual in range(self.population_size):
            chromosome = Chromosome(self.chromosome_lengths)
            # chromosome.fitness = self.eval_objective_func(chromosome)
            population.append(chromosome)
        return population

    # Ta metoda zwraca ocenę osobnika - chromosome
    def eval_objective_func(self, chromosome):
        lower = 0
        upper = self.bits_per_arg - 1
        arguments = []
        while upper < self.chromosome_lengths:
            argument = chromosome.decode(lower, upper, self.aoi)
            arguments.append(argument)
            lower += self.bits_per_arg
            upper += self.bits_per_arg
        # po pętli zwiększamy oba o self.bits_per_arg

        # mamy już listę argumentów tej funkcji, teraz zwrócić jej wartość
        # w funkcji zastgosuejmy args żeby przyjmowałą dowolną ilość argumentów
        chromosome.arguments = arguments
        chromosome.fitness = self.objective_function(*arguments)
        # return chromosome.fitness
        # return self.objective_function(arguments)

        # w chromosome dostajemy array któy musimy podzielić na tyle częsci ile jest w obj_fuc_num_args
        # (możemy juz skorzystac z self.bits_per_arg)- wyliczonej w konstruktorze 

        # każdą z tych części dekodujemy wtedy z osobna - otrzymując argumenty funckji (np. x1, x2)
        # do dekodowania będziemy się przesuwać kolejno p self.bits_per_arg bitów i self.aoi też mamy podane
        # i liczym wartość tej funckji dzięki self.objective_function (ta funkcja została podana do góry zadania)
        # returnujemy wartość

    def evaluation(self):
        for chromosome in self.population:
            self.eval_objective_func(chromosome)

    def tournament_selection(self):
        parents = []
        # tournaments_number = int(self.population_size / self.tournament_size)
        # for iteration in range(self.population_size):
        # # for iteration in range(tournaments_number):
        #     tournament = random.sample(self.population, self.tournament_size)
        #     winner = min(tournament, key=lambda chromosome: chromosome.fitness)
        #     parents.append(winner)

        for individual in self.population:
            candidates = [individual]
            candidates.extend(random.sample(self.population, self.tournament_size - 1))
            winner = min(candidates, key=lambda chromosome: chromosome.fitness)
            parents.append(winner)
        # z wszystkich osobników wybieram tylu ile w self.tournament size
        # porównuję ich na podstawie funkcji celu, zwycięzce zapisuje
        # na koniec otrzymuję listę zwycięzców - parents, którą returnujemy
        return parents

    def reproduce(self, parents):
        descendants = []
        # rounds = int(len(parents) / 2)
        # tu też do końca nie wiem jak obrać ilość przeprowadzanych rund
        # for round in range(self.population_size):
        # # for round in range(rounds):
        #     new_parents = random.sample(parents, 2)
        #     if np.random.rand() < self.crossover_probability:
        #         descendant = new_parents[0].crossover(new_parents[1])
        #         descendant.mutation(self.mutation_probability)
        #         descendants.append(descendant)
        #         # może się okazać, żę jednak obu potomków mam przekazać dalej, jeszcze nie wiem
        #     else:
        #         descendant = min(new_parents, key=lambda chromosome: chromosome.fitness)
        #         descendant.mutation(self.mutation_probability)
        #         descendants.append(descendant)     

        for individual in self.population:
            partner = random.choice(self.population)
            if np.random.rand() < self.crossover_probability:
                descendant = individual.crossover(partner)
                descendant.mutation(self.mutation_probability)
                descendants.append(descendant)
                # może się okazać, żę jednak obu potomków mam przekazać dalej, jeszcze nie wiem
            else:
                descendant = min([individual, partner], key=lambda chromosome: chromosome.fitness)
                descendant.mutation(self.mutation_probability)
                descendants.append(descendant)
                # descendant1 = new_parents[0]
                # descendant1.mutation(self.mutation_probability)
                # descendant2 = new_parents[1]
                # descendant2.mutation(self.mutation_probability)
                
                # descendant1 = new_parents[0].mutation(self.mutation_probability)
                # descendant2 = new_parents[1].mutation(self.mutation_probability)
                
                # descendants.extend([descendant1, descendant2])

        # generuje losowe prawdopodbienstwo i jezeli mniejsze od croosover to przeprowadzam crossover i potomka wrzucam
        # jezeli nie to rodzice trafiaja do nowego pokolenia
        # zwracam nowe pokolenie
        return descendants

    def plot_func(self, trace):
        X = np.arange(-2, 3, 0.05)
        Y = np.arange(-4, 2, 0.05)
        X, Y = np.meshgrid(X, Y)
        Z = 1.5 - np.exp(-X ** (2) - Y ** (2)) - 0.5 * np.exp(-(X - 1) ** (2) - (Y + 2) ** (2))
        plt.figure()
        plt.contour(X, Y, Z, 10)
        cmaps = [[ii / len(trace), 0, 0] for ii in range(len(trace))]
        plt.scatter([x[0] for x in trace], [x[1] for x in trace], c=cmaps)
        plt.show()

    def run(self):
        #  ileś podanych w num_step razy przeprowadzamy ren algorytm
        trace = []
        for step in range(self.num_steps):
            # dla każdego z populacji przeprowadzam wartościowanie
            self.evaluation()

            # dla każdego pokolenia szukamy w nim najlepszego osobnika
            best = min(self.population, key=lambda chromosome: chromosome.fitness)
            self.best_of_generation.append((best, best.arguments, best.fitness))
            trace.append(best.arguments)
            print(best.arguments)
            print(best.fitness)
            print('\n')

            # przeprowadzamy selekcję turniejową (turnieje, w każdym turnieju z kilku losowych wynieramy najlepszego, któy będzie w reprodukcji)
            parents = self.tournament_selection()

            # przeprowadzamy reprodukcję  (albo krzyżujemy albo przechodzą do następnego pokolenia), mutujemy
            descendants = self.reproduce(parents)

            # deklarujemt nowe pokolenie
            self.population = descendants
        self.plot_func(trace)


def objective_function(*args):
    X = args[0]
    Y = args[1]
    return 1.5 - np.exp(-X**2 - Y**2) - 0.5 * np.exp(-(X - 1)**2 - (Y + 2)**2)


chrom_len = 8
arguments = 2
aoi = [0, 1]
pop = 1000
t=2
mut=0.05
cros=0.8
num=30

gen = GeneticAlgorithm(chrom_len, arguments, objective_function, aoi, pop, t, mut, cros, num)
gen.run()


chrom_len2 = 32
arguments2 = 2
aoi2 = [-20, 20]
pop2 = 1000
t2=2
mut2=0.05
cros2=0.8
num2=30

gen2 = GeneticAlgorithm(chrom_len2, arguments2, objective_function, aoi2, pop2, t2, mut2, cros2, num2)
gen2.run()
# WYKRES TO WIZUALIZACJA
# trace to wypisanie najlepszych osobników w każdym pokoleniu
# minimum to współrzedne ostatniego osobnika

# PYTANIA:
# 1. Na jakim etapie powinna zostać przeprowadzona mutacja
# czy lepsze z dzieci podaczas krzyżowania --- ja założyłam, że wybierane jest losowe dziecko
# czy obu rodziców dalej ---- założyłam, że by populacja pozostała stała, dalej przechodzi jedynie "lepszy z rodziców"

# ile powtórzeń dla tunrieju i ile dla reprodukcji --- 
# założyłam, że tyle ile wynosi rozmiar populacji, dla turnieju musi być tyle, by uzyskać
# wystarczającą ilość rodziców, podobnie w reproduckji, skoro z dwóch rodziców otrzymujemy jedno dziecko, proces taki musi powtórzyć się tyle razy 
# ile wynosi rozmiar popualcji. Alternatywnie zastanawiałam się nad przeprowadzeniem N/2 ilości cykli w reprodukcji, wtedy to w wyniku jednego cyklu 
# dwóch rodziców otrzymałoby dwóch potomków lub geny obojgu przeszłyby do następnego pokolenia. Zdecydowałam jednak, że moje rozwiązanie zapewni większą
# różnorodność - pozwoli na częstsze zkrzyżowanie różnych osobników.


# WNIOSEK DO WYKRESU
# Wizualizacja przedstawiająca odnalezienie minimum przy pomocy algorytmu genetycznego pokazuje, że punkty zbliżające się do wartośći najmniejszej są porozrzucane
# po całej planszy. Możemy zauważyć zależności (przedstawioną tutaj poprzez coraz jaśniejszy kolor punktów), że wraz z kolejnymi iteracjami algorytmu punkty 
# przybliżają się do minimum, jednak jest to jedyna cecha, która pozwala na umiejscowienie ich względem siebie. Choć każdy następny znajduje się bliżej minima, na wykresie
# nie możemy odnaleźć żadnego innego powiązania pomiędzy punktami - nie układają się on w linię, bądź krzywą, która stopniowo przybliża nas do wartości najmniejszej.  
# Sytuacja wyglądała inaczej w przypadku przeprowadzonej wcześniej minimalizacji metodą gradientu prostego. Tam, przy odpowiednim dobraniu współcznnika alfa,
# mogliśmy zauważyć, że dane punkty tworzą pewien określony ślad, rozpoczynający się w punkcie początkowym i kończący się w minimum. 