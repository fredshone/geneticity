import pandas as pd
import numpy as np
from scipy.spatial.distance import cityblock
import random
import matplotlib.pyplot as plt


class CityTemplate:
    df = None
    size = None
    name = None
    verbose = None

    def __init__(self, name="City", size=(10, 10), verbose=True):
        self.name = name
        self.size = size
        self.verbose = verbose

    def load_csv(self, path="city1.csv", verbose=True):
        self.df = pd.read_csv(path, index_col=0)
        assert self.size[0] * self.size[1] == len(self.df), 'csv dimensions do not match size'
        if verbose:
            print(self.df.head())
            print(self.df.tail())

    def load_test_city(self, group_size=1):
        """
        Make city dataframe
        """
        headers = ['id', 'type', 'people', 'jobs', 'public']
        output = []
        ref = 1
        for y in range(self.size[0]):
            for x in range(self.size[1]):
                line = [ref]
                # types = park, commercial, residential
                if y < group_size and x < group_size:
                    line += ['residential', 80, 20, 0]
                elif y >= self.size[0] - group_size and x >= self.size[1] - group_size:
                    line += ['commercial', 0, 100, 0]
                else:
                    line += ['park', 0, 0, 100]
                output.append(line)
                ref += 1
        df = pd.DataFrame(output, columns=headers)
        self.df = df.set_index('id')
        if self.verbose:
            print(self.df.head())

    def load_centric_city(self, centre_radius=2, parks=0.15):
        """
        Make city dataframe
        """
        headers = ['id', 'type', 'people', 'jobs', 'public']
        output = []
        ref = 1
        max_radius = self.size[0] / 2
        half_array = (np.array(self.size) / 2) - 0.5
        for y in range(self.size[0]):
            for x in range(self.size[1]):
                line = [ref]
                dist = max(abs(half_array - np.array((float(y), float(x)))))
                # types = park, commercial, residential
                if np.random.choice([True, False], p=[parks, 1 - parks]):
                    line += ['park', 0, 0, 100]
                elif dist <= centre_radius:
                    line += ['centre', 10, 80, 10]
                else:
                    fraction = (dist - centre_radius) / (max_radius - centre_radius)
                    people = 10 + (fraction * 80)
                    jobs = random.uniform(10, 90 - people)
                    public = 100 - people - jobs
                    line += ['residential', people, jobs, public]
                output.append(line)
                ref += 1
        df = pd.DataFrame(output, columns=headers)
        self.df = df.set_index('id')
        if self.verbose:
            print(self.df.head())


class City:
    verbose = False
    weights = {'ptu_weight': 1,
               'pdu_weight': 1,
               'ewu_weight': 1,
               'eau_weight': 1,
               'p_weight': 1,
               'e_weight': 1
               }
    ptu_weight = weights['ptu_weight']
    pdu_weight = weights['pdu_weight']
    ewu_weight = weights['ewu_weight']
    eau_weight = weights['eau_weight']
    p_weight = weights['p_weight']
    e_weight = weights['e_weight']

    score = None
    population_score = None
    employer_score = None
    population_util_array = None
    employer_util_array = None
    scored = False

    def __init__(self, template):
        self.template = template
        self.df = template.df
        self.size = template.size
        self.array = self.df.index.values.reshape(self.size)
        self.name = template.name
        assert len(self.df) == self.size[0] * self.size[1], "dimensions {} don't match input df".format(self.size)

    def get_seeds(self, shuffle=False):
        copy = City(self.template)
        if shuffle == 'rows':
            copy.shuffle_rows()
        elif shuffle:
            copy.shuffle()
        return copy

    def shuffle(self):
        self.array = self.array.flatten()
        np.random.shuffle(self.array)
        self.array = self.array.reshape(self.size[0], self.size[1])

    def shuffle_rows(self):
        self.array = np.random.permutation(self.array)

    def calculate_block_utility(self, block_location):
        ptu = 0  # resident travel utility
        pdu = 0  # resident density utility
        ewu = 0  # employer workforce utility
        eau = 0  # employer agglomeration utility
        block_id = self.array[block_location]
        stats = self.df.loc[block_id, :]
        for location, block_id in np.ndenumerate(self.array):
            block_stats = self.df.loc[block_id, :]
            if block_stats.type == 'park': continue  # park - no util
            dist = cityblock(block_location, location)
            if dist == 0: continue  # Same block
            if dist <= 6:  # resident work travel util - disutil for distance to work
                ptu += (1 - (dist / 6)) * block_stats.jobs
            if dist < 2:  # resident density util - disutil for high density (max = 900)
                pdu += 100 - block_stats.people - block_stats.jobs
            ewu += block_stats.people / dist  # commercial - util for lots of close pop
            eau += block_stats.jobs / (dist ** 1.2)  # commercial - util for lots of close commercial
        pop_utility = ((self.ptu_weight * ptu) + (self.pdu_weight * pdu)) * stats.people * self.p_weight
        employer_utility = ((self.ewu_weight * ewu) + (self.eau_weight * eau)) * stats.jobs * self.e_weight
        return pop_utility, employer_utility

    def calc_utility(self):
        self.population_util_array = np.zeros_like(self.array)
        self.employer_util_array = np.zeros_like(self.array)
        for index, _block in np.ndenumerate(self.array):
            self.population_util_array[index], self.employer_util_array[index] = self.calculate_block_utility(index)
        self.population_score = np.sum(self.population_util_array)
        self.employer_score = np.sum(self.employer_util_array)
        self.score = self.population_score + self.employer_score

    def mutate(self):
        a = (random.randrange(0, self.size[0], 1), random.randrange(0, self.size[1], 1))
        b = (random.randrange(0, self.size[0], 1), random.randrange(0, self.size[1], 1))
        self.array[a[0], a[1]], self.array[b[0], b[1]] = self.array[b[0], b[1]], self.array[a[0], a[1]]

    def debug(self):
        self.verbose = True

    def show_array(self):
        print(self.array)

    # Plotting ----------------------------------

    def get_type(self, name):
        use = self.df.loc[name ,'type']
        if use == 'park':
            return use, 100
        elif use == 'commercial':
            return use, self.df.loc[name, 'jobs']
        elif use == 'residential':
            return use, self.df.loc[name, 'people']
        else:
            print("Unknown type")

    def get_density_array(self, choice='people'):
        copy = np.zeros_like(self.array)
        for location, key in np.ndenumerate(self.array):
            value = self.df.loc[key, choice]
            copy[location] = value
        return copy

    def get_use(self, name, plot='jobs'):
        return self.df.loc[name, plot]

    def draw(self, plot='land_use'):
        figure, axes = plt.subplots(1, 1, figsize=(5, 5))
        if plot == 'population_utility':
            intensity = -self.population_util_array
            cmap = 'seismic'
        elif plot == 'employer_utility':
            intensity = -self.employer_util_array
            cmap = 'seismic'
        elif plot == 'land_use':
            get_intensity = np.vectorize(self.get_use)
            intensity = -get_intensity(self.array, 'jobs')
            cmap = 'RdYlGn'
        elif plot == 'green_space':
            get_intensity = np.vectorize(self.get_use)
            intensity = get_intensity(self.array, 'public')
            cmap = 'Greens'
        else:
            raise Exception('Unknown plot called')
        axes.imshow(intensity, interpolation="nearest", cmap=plt.get_cmap(cmap))
        axes = plt.gca()
        # Minor ticks
        axes.set_xticks(np.arange(-.5, self.size[1], 1), minor=True)
        axes.set_yticks(np.arange(-.5, self.size[0], 1), minor=True)

        # Gridlines based on minor ticks
        axes.grid(which='minor', color='w', linestyle='-', linewidth=2)

        for tic in axes.xaxis.get_major_ticks():
            tic.tick1On = tic.tick2On = False
            tic.label1On = tic.label2On = False
        for tic in axes.yaxis.get_major_ticks():
            tic.tick1On = tic.tick2On = False
            tic.label1On = tic.label2On = False
        for tic in axes.xaxis.get_minor_ticks():
            tic.tick1On = tic.tick2On = False
            tic.label1On = tic.label2On = False
        for tic in axes.yaxis.get_minor_ticks():
            tic.tick1On = tic.tick2On = False
            tic.label1On = tic.label2On = False
        axes.spines['top'].set_visible(False)
        axes.spines['right'].set_visible(False)
        axes.spines['bottom'].set_visible(False)
        axes.spines['left'].set_visible(False)

        axes.set_title(plot)

        plot_title = self.name
        figure.suptitle(plot_title)
        plt.show()

    def diagnose(self):
        if not self.scored:
            self.calc_utility()
            self.scored = True
        pop_density = self.get_density_array('people')
        job_density = self.get_density_array('jobs')
        figure, axes = plt.subplots(1, 4, figsize=(20, 3))
        plots = ('land_use', 'green_space', 'population_utility', 'employer_utility')
        for index, plot in enumerate(plots):
            if plot == 'population_utility':
                intensity = -self.population_util_array
                cmap = 'seismic'
            elif plot == 'employer_utility':
                intensity = -self.employer_util_array
                cmap = 'seismic'
            elif plot == 'land_use':
                get_intensity = np.vectorize(self.get_use)
                intensity = -get_intensity(self.array, 'jobs')
                cmap = 'RdYlGn'
            elif plot == 'green_space':
                get_intensity = np.vectorize(self.get_use)
                intensity = get_intensity(self.array, 'public')
                cmap = 'Greens'
            else:
                raise Exception('Unknown plot called')

            axes[index].imshow(intensity, interpolation="nearest", cmap=plt.get_cmap(cmap))
            # Minor ticks
            axes[index].set_xticks(np.arange(-.5, self.size[1], 1), minor=True);
            axes[index].set_yticks(np.arange(-.5, self.size[0], 1), minor=True);

            # Gridlines based on minor ticks
            axes[index].grid(which='minor', color='w', linestyle='-', linewidth=2)

            for tic in axes[index].xaxis.get_major_ticks():
                tic.tick1On = tic.tick2On = False
                tic.label1On = tic.label2On = False
            for tic in axes[index].yaxis.get_major_ticks():
                tic.tick1On = tic.tick2On = False
                tic.label1On = tic.label2On = False
            for tic in axes[index].xaxis.get_minor_ticks():
                tic.tick1On = tic.tick2On = False
                tic.label1On = tic.label2On = False
            for tic in axes[index].yaxis.get_minor_ticks():
                tic.tick1On = tic.tick2On = False
                tic.label1On = tic.label2On = False
            axes[index].spines['top'].set_visible(False)
            axes[index].spines['right'].set_visible(False)
            axes[index].spines['bottom'].set_visible(False)
            axes[index].spines['left'].set_visible(False)

            axes[index].set_title(plot)

        plot_title = self.name
        figure.suptitle(plot_title)
        plt.show()


class Generation:
    verbose = False
    name = None
    scores = None
    seed_city = None
    children = None
    best_city = None
    best_score = None
    population_score = None
    employer_score = None

    def __init__(self, size, breeders, killed, mutations=2):
        assert breeders <= size
        assert killed <= size
        self.size = size
        self.breeders = breeders
        self.killed = killed
        self.mutations = mutations
        self.cities = []
        self.number = 0

    def seed(self, seed_city, shuffle=True):
        self.seed_city = seed_city
        self.name = seed_city.name
        for i in range(self.size):
            self.cities.append(seed_city.get_seeds(shuffle))

    def copy(self):
        new_generation = Generation(self.size, self.breeders, self.killed, self.mutations)
        new_generation.cities = self.cities
        new_generation.number = self.number + 1
        new_generation.scores = self.scores
        return new_generation

    def breed(self):
        odds_of_breeding = self.scores[:self.breeders]
        self.children = []
        for c in range(self.killed):
            mother = random.choices(self.cities[:self.breeders], weights=odds_of_breeding)[0]
            father = random.choices(self.cities[:self.breeders], weights=odds_of_breeding)[0]
            child = City(mother.template)
            child.array = np.zeros_like(mother.array)
            for index, name in enumerate(mother.df.index):
                if index < child.array.size / 2:
                    position = np.where(mother.array == name)
                else:
                    position = np.where(father.array == name)

                if child.array[position]:
                    filtered = np.where(np.equal(child.array, 0))
                    position = random.choice(list(zip(filtered[0], filtered[1])))

                child.array[position] = name
            self.children.append(child)

    def breed2(self):
        odds_of_breeding = self.scores[:self.breeders]
        self.children = []
        for c in range(self.killed):
            mother = random.choices(self.cities[:self.breeders], weights=odds_of_breeding)[0]
            father = random.choices(self.cities[:self.breeders], weights=odds_of_breeding)[0]
            child = City(mother.template)
            child.array = np.zeros_like(mother.array)
            for index, name in enumerate(mother.df.index):
                if index % 2 == 0:
                    position = np.where(mother.array == name)
                else:
                    position = np.where(father.array == name)

                if child.array[position]:
                    filtered = np.where(np.equal(child.array, 0))
                    position = random.choice(list(zip(filtered[0], filtered[1])))

                child.array[position] = name
            self.children.append(child)

    def kill_weakest(self):
        for city in range(self.killed):
            del self.cities[-1]

    def mutate_children(self):
        for child in self.children:
            for m in range(self.mutations):
                child.mutate()

    def add_children(self):
        self.cities.extend(self.children)
        assert len(self.cities) == self.size

    def score(self):
        for city in self.cities:
            city.calc_utility()

    def rank_generation(self):
        assert self.cities[0].score is not None, 'Check if cities have been scored yet'
        self.scores = []
        self.cities.sort(key=lambda x: x.score, reverse=True)
        for city in self.cities:
            score = city.score
            self.scores.append(score)
        if self.verbose:
            print(self.scores)
        self.best_city = self.cities[0]
        self.best_score = self.best_city.score
        self.population_score = self.best_city.population_score
        self.employer_score = self.best_city.employer_score

    # Plotting ----------------------------------

    def draw_generation(self, max_plots=12):
        columns = min(self.size, max_plots)
        figure, axes = plt.subplots(1, columns, figsize=(20, 2))

        for index, city in enumerate(self.cities[:columns]):
            get_intensity = np.vectorize(city.get_use)
            intensity = -get_intensity(city.array, 'jobs')
            axes[index].imshow(intensity, interpolation="nearest", cmap=plt.get_cmap('RdYlGn'))
            axes[index].set_axis_off()
            subplot_title = "City " + str(index)
            axes[index].set_title(subplot_title)

        plot_title = "Generation " + str(self.number)
        figure.suptitle(plot_title)
        plt.show()

    def debug(self):
        self.verbose = True


class CityEvolution:
    verbose = True
    name = None
    seed_generation = None
    generations = None
    early_stop = False
    best_cities = []
    best_scores = []
    population_scores = []
    employer_scores = []

    def __init__(self, name='City Evolution', generations=20):
        self.name = name
        self.generations = generations
        if self.verbose:
            print("Setting up City Evolution with {} generations".format(self.generations))

    def seed(self, seed):
        self.best_cities = []
        self.best_scores = []
        self.population_scores = []
        self.employer_scores = []
        self.seed_generation = seed
        if self.verbose:
            print('Seeded with generation "{}" of population {}'.format(seed.name, seed.size))

    def run(self):
        if self.verbose:
            print("Starting city evolution")
            if self.early_stop:
                print("Early stopping enables")
        self.seed_generation.score()
        self.seed_generation.rank_generation()
        self.seed_generation.draw_generation()
        self.best_scores.append(self.seed_generation.best_score)
        self.population_scores.append(self.seed_generation.population_score)
        self.employer_scores.append(self.seed_generation.employer_score)
        self.best_cities.append(self.seed_generation.best_city)
        last_generation = self.seed_generation
        for generation_num in range(self.generations):
            new_generation = last_generation.copy()
            new_generation.breed()
            new_generation.kill_weakest()
            new_generation.mutate_children()
            new_generation.add_children()
            new_generation.score()
            new_generation.rank_generation()
            self.best_scores.append(new_generation.best_score)
            self.population_scores.append(new_generation.population_score)
            self.employer_scores.append(new_generation.employer_score)
            self.best_cities.append(new_generation.best_city)
            new_generation.draw_generation()
            last_generation = new_generation

    def plot_scores(self):
        plt.plot(self.best_scores)
        plt.plot(self.population_scores)
        plt.plot(self.employer_scores)
        plt.title("Most Fit Trend")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.show()

    def diagnose(self):
        self.best_cities[0].diagnose()
        self.best_cities[-1].diagnose()



