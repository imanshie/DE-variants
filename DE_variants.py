import numpy as np
import random
import math
from copy import deepcopy
from Beamforming import Problem, Report
import InputOutput


class DE:
    @staticmethod
    def static(D, BOUNDS, GENE_NO, MAX_NFE, PN_INIT, PN_MIN):
        DE.MAX_NFE, DE.PN_INIT, DE.D, DE.BOUNDS, DE.GENE_NO = MAX_NFE, PN_INIT, D, BOUNDS, GENE_NO
        DE.PN_MIN = PN_MIN
        DE.POP_INIT = DE.create_population()
        results = [Problem.evaluation(vec) for vec in DE.POP_INIT]
        DE.POP_INIT = [vec['vec'] for vec in results]
        DE.POP_INIT_FITNESS = [vec['fitness'] for vec in results]
        DE.POP_INIT_CONSTRAINTS = [vec['constraints'] for vec in results]
        DE.POP_INIT_CONSTRAINTS_VIOLATION_NUM = [sum(1 for constraint in vec_constraints if constraint > 0) for vec_constraints in DE.POP_INIT_CONSTRAINTS]
        DE.POP_INIT_FEASIBILITY = [True if constraints_violation_num == 0 else False for constraints_violation_num in DE.POP_INIT_CONSTRAINTS_VIOLATION_NUM]

    @classmethod
    def create_population(cls):
        return \
            [
                [
                    [
                        low + (random.uniform(0, 1) * (high - low))
                        for low, high in zip(DE.BOUNDS['low'], DE.BOUNDS['high'])
                    ]
                    for _ in range(DE.D)
                ]
                for _ in range(DE.PN_INIT)
            ]

    def initialize_population(self):
        self.PN = DE.PN_INIT
        self.pop = deepcopy(DE.POP_INIT)
        self.pop_fitness = deepcopy(DE.POP_INIT_FITNESS)
        self.pop_constraints = deepcopy(DE.POP_INIT_CONSTRAINTS)
        self.pop_feasibility = deepcopy(DE.POP_INIT_FEASIBILITY)
        self.pop_constraints_violation_num = deepcopy(DE.POP_INIT_CONSTRAINTS_VIOLATION_NUM)

    def set_trial_to_target(self):
        self.pop_constraints[self.ind_trgt_vec] = self.trial_constraints
        self.pop_fitness[self.ind_trgt_vec] = self.trial_fitness
        self.pop_feasibility[self.ind_trgt_vec] = self.trial_feasibility
        self.pop_constraints_violation_num[self.ind_trgt_vec] = self.trial_constraints_violation_num
        self.pop[self.ind_trgt_vec] = self.trial

    def check_donor_bounds(self):
        for gene_donor, gene_target in zip(self.donor, self.pop[self.ind_trgt_vec]):
            for ind, low, high in zip(range(self.PN), DE.BOUNDS['low'], DE.BOUNDS['high']):
                if gene_donor[ind] < low:
                    gene_donor[ind] = (gene_target[ind] + low) / 2
                elif gene_donor[ind] > high:
                    gene_donor[ind] = (gene_target[ind] + high) / 2
        return self.donor

    def recombine(self):
        CR = self.get_CR()
        rand_j = random.choice(list(range(DE.D)))
        self.trial = []
        for j in range(DE.D):
            if random.uniform(0, 1) <= CR or j == rand_j:
                self.trial.append(self.donor[j])
            else:
                self.trial.append(self.pop[self.ind_trgt_vec][j])

    def evaluation_trial(self):
        result = Problem.evaluation(self.trial)
        self.trial = result['vec']
        self.trial_constraints = result['constraints']
        self.trial_fitness = result['fitness']
        self.trial_constraints_violation_num = sum(1 for constraint in self.trial_constraints if constraint > 0)
        self.trial_feasibility = True if self.trial_constraints_violation_num == 0 else False

    def is_trial_better(self):
        return True \
            if (self.trial_feasibility and self.pop_feasibility[self.ind_trgt_vec]) \
               and (self.trial_fitness <= self.pop_fitness[self.ind_trgt_vec]) \
               or (self.trial_feasibility and self.pop_feasibility[self.ind_trgt_vec] is not True) \
               or self.trial_constraints_violation_num <= self.pop_constraints_violation_num[self.ind_trgt_vec] \
            else False

    def selection(self):
        if self.is_trial_better():
            self.set_trial_to_target()

    def get_sorted_pop_indexes(self):
        feasible_vecs_indexes, feasible_pop_fitnesses = [], []
        infeasible_vecs_indexes, infeasible_pop_violations_sum = [], []
        for ind in range(self.PN):
            if self.pop_feasibility[ind]:
                feasible_vecs_indexes.append(ind)
                feasible_pop_fitnesses.append(self.pop_fitness[ind])
            else:
                infeasible_vecs_indexes.append(ind)
                infeasible_pop_violations_sum.append(self.pop_constraints_violation_num[ind])

        temp_1 = np.argsort(feasible_pop_fitnesses)
        temp_2 = np.argsort(infeasible_pop_violations_sum)
        indexes = [feasible_vecs_indexes[index] for index in temp_1] + [infeasible_vecs_indexes[index] for index in temp_2]
        return indexes

    def rand_mutation(self):
        F = self.get_F()

        candidates = list(range(self.PN))
        candidates.remove(self.ind_trgt_vec)
        rand_index_1 = random.choice(candidates)
        candidates.remove(rand_index_1)
        rand_index_2 = random.choice(candidates)
        candidates.remove(rand_index_2)
        rand_index_3 = random.choice(candidates)

        self.donor = [
                [
                    self.pop[rand_index_1][indiv_index][gene_index] +
                    F * (self.pop[rand_index_2][indiv_index][gene_index] - self.pop[rand_index_3][indiv_index][gene_index])
                    for gene_index in range(DE.GENE_NO)
                ]
                for indiv_index in range(DE.D)
        ]
        return self.check_donor_bounds()

    def best_mutation(self):
        F = self.get_F()

        candidates = list(range(self.PN))
        candidates.remove(self.ind_trgt_vec)
        rand_index_1 = random.choice(candidates)
        candidates.remove(rand_index_1)
        rand_index_2 = random.choice(candidates)

        self.donor = [
                [
                    self.pop[self.best_vec_index][indiv_index][gene_index] +
                    F * (self.pop[rand_index_1][indiv_index][gene_index] - self.pop[rand_index_2][indiv_index][gene_index])
                    for gene_index in range(DE.GENE_NO)
                ]
                for indiv_index in range(DE.D)
        ]
        return self.check_donor_bounds()

    def current_to_best_mutation(self):
        F = self.get_F()
        candidates = list(range(self.PN))
        candidates.remove(self.ind_trgt_vec)
        rand_index_1 = random.choice(candidates)
        candidates.remove(rand_index_1)
        rand_index_2 = random.choice(candidates)

        self.donor = [
                [
                    self.pop[self.ind_trgt_vec][indiv_index][gene_index] +
                    F * (self.pop[self.best_vec_index][indiv_index][gene_index] - self.pop[self.ind_trgt_vec][indiv_index][gene_index]) +
                    F * (self.pop[rand_index_1][indiv_index][gene_index] - self.pop[rand_index_2][indiv_index][gene_index])
                    for gene_index in range(DE.GENE_NO)
                ]
                for indiv_index in range(DE.D)
        ]
        return self.check_donor_bounds()

    def current_to_pbest_mutation(self):
        F = self.get_F()

        candidates = list(range(self.PN))
        candidates.remove(self.ind_trgt_vec)
        rand_index_1 = random.choice(candidates)
        candidates.remove(rand_index_1)
        rand_index_2 = random.choice(candidates)

        rand_best_index = random.choice(list(self.best_vecs_indexes))

        self.donor = [
                [
                    self.pop[self.ind_trgt_vec][indiv_index][gene_index] +
                    F * (self.pop[rand_best_index][indiv_index][gene_index] - self.pop[self.ind_trgt_vec][indiv_index][gene_index]) +
                    F * (self.pop[rand_index_1][indiv_index][gene_index] - self.pop[rand_index_2][indiv_index][gene_index])
                    for gene_index in range(DE.GENE_NO)
                ]
                for indiv_index in range(DE.D)
             ]
        return self.check_donor_bounds()

    def current_to_pbest_A_mutation(self):
        F = self.get_F()

        candidates = list(range(self.PN))
        candidates.remove(self.ind_trgt_vec)
        rand_index = random.choice(candidates)

        pop_A = self.pop + self.A
        candidates = list(range(len(pop_A)))
        candidates.remove(self.ind_trgt_vec)
        candidates.remove(rand_index)
        rand_index_pop_A = random.choice(candidates)

        rand_best_index = random.choice(list(self.best_vecs_indexes))

        self.donor = [
                [
                    self.pop[self.ind_trgt_vec][indiv_index][gene_index] +
                    F * (self.pop[rand_best_index][indiv_index][gene_index] - self.pop[self.ind_trgt_vec][indiv_index][gene_index]) +
                    F * (self.pop[rand_index][indiv_index][gene_index] - pop_A[rand_index_pop_A][indiv_index][gene_index])
                    for gene_index in range(DE.GENE_NO)
                ]
                for indiv_index in range(DE.D)
            ]
        return self.check_donor_bounds()


class Tuning(DE):
    ALGO_TYPE = "Parameter-Tuning"

    @staticmethod
    def static(F=0.5, CR=0.9):
        Tuning.F = F
        Tuning.CR = CR

    def get_F(self): return Tuning.F

    def get_CR(self): return Tuning.CR

    def feedback(self):
        pass


class DERand1(Tuning):
    ALGO_NAME = "DE/rand/1"
    ALGO_FOLDER_NAME = "DE_rand_1"
    MUTATION_TYPE = "DE/rand/1"

    def __init__(self):
        self.initialize_population()
        self.NFE = 0

    def mutation(self):
        self.donor = self.rand_mutation()


class DECurrentToBest1(Tuning):
    ALGO_NAME = "DE/current-to-best/1"
    ALGO_FOLDER_NAME = "DE_current-to-best_1"
    MUTATION_TYPE = "DE/current-to-best/1"

    def __init__(self):
        self.initialize_population()
        self.NFE = 0

    def mutation(self):
        self.best_vec_index = int(self.sorted_pop_indexes[0])
        self.donor = self.current_to_best_mutation()


class DEBest1(Tuning):
    ALGO_NAME = "DE/best/1"
    ALGO_FOLDER_NAME = "DE_best_1"
    MUTATION_TYPE = "DE/best/1"

    def __init__(self):
        self.initialize_population()
        self.NFE = 0

    def mutation(self):
        self.best_vec_index = int(self.sorted_pop_indexes[0])
        self.donor = self.best_mutation()


class SelfAdaptive(DE):
    ALGO_TYPE = "Self-Adaptive-Parameter-Control"

    def feedback(self):
        pass


class jDE(SelfAdaptive):
    ALGO_NAME = "jDE"
    ALGO_FOLDER_NAME = "jDE"
    MUTATION_TYPE = "DE/rand/1"

    def __init__(self, Fl=0.1, Fu=0.9, T1=0.1, T2=0.1):
        self.Fl, self.Fu, self.T1, self.T2 = Fl, Fu, T1, T2
        self.initialize_population()
        self.NFE = 0
        self.F = [0.5] * self.PN
        self.CR = [0.9] * self.PN

    def get_F(self):
        if random.uniform(0, 1) < self.T1:
            self.F[self.ind_trgt_vec] = self.Fl + random.uniform(0, 1) * self.Fu
        return self.F[self.ind_trgt_vec]

    def get_CR(self):
        if random.uniform(0, 1) < self.T2:
            self.CR[self.ind_trgt_vec] = random.uniform(0, 1)
        return self.CR[self.ind_trgt_vec]

    def mutation(self):
        self.donor = self.rand_mutation()


class Adaptive(DE):
    ALGO_TYPE = "Adaptive-Parameter-Control"


class JADEWO(Adaptive):
    ALGO_NAME = "JADE-without-Archive"
    ALGO_FOLDER_NAME = "JADE_without_Archive"
    MUTATION_TYPE = "DE/current-to-pbest"

    def __init__(self, pbest, MF=0.5, MCR=0.5, C=0.1):
        self.initialize_population()
        self.pbest, self.MF, self.MCR, self.C = pbest, MF, MCR, C
        self.best_vecs_len = math.ceil(self.pbest * DE.PN_INIT)
        self.SF, self.SCR = [], []
        self.NFE = 0

    def get_F(self):
        self.F = Rand.cauchy(self.MF, 0.1)
        if self.F > 1:
            self.F = 1
        while self.F <= 0 or self.F > 1:
            self.F = Rand.cauchy(self.MF, 0.1)
        return self.F

    def get_CR(self):
        self.CR = np.random.normal(self.MCR, 0.1)
        if self.CR > 1:
            self.CR = 1
        elif self.CR < 0:
            self.CR = 0
        return self.CR

    def mutation(self):
        self.best_vecs_indexes = self.sorted_pop_indexes[:self.best_vecs_len]
        self.donor = self.current_to_pbest_mutation()

    def selection(self):
        if self.is_trial_better():
            if self.pop_fitness[self.ind_trgt_vec] != self.trial_fitness:
                self.SF.append(self.F)
                self.SCR.append(self.CR)
            self.set_trial_to_target()

    def feedback(self):
        self.MF = (1 - self.C) * self.MF + self.C * Mean.lehmer(self.SF)
        self.MCR = (1 - self.C) * self.MCR + self.C * Mean.arithmetic(self.SCR)
        self.SF, self.SCR, self.trial_fitness = [], [], []


class JADEW(Adaptive):
    ALGO_NAME = "JADE-with-Archive"
    ALGO_FOLDER_NAME = "JADE_with_Archive"
    MUTATION_TYPE = "DE/current-to-pbest-Archive"

    def __init__(self, AN, pbest=0.05, MF=0.5, MCR=0.5, C=0.1):
        self.initialize_population()
        self.A = []
        self.pbest, self.MF, self.MCR, self.C = pbest, MF, MCR, C
        self.best_vecs_len = int(math.ceil(self.pbest * DE.PN_INIT))
        self.SF, self.SCR = [], []
        self.NFE = 0
        self.AN = AN

    def get_F(self):
        self.F = Rand.cauchy(self.MF, 0.1)
        if self.F > 1:
            self.F = 1
        while self.F <= 0 or self.F > 1:
            self.F = Rand.cauchy(self.MF, 0.1)
        return self.F

    def get_CR(self):
        self.CR = np.random.normal(self.MCR, 0.1)
        if self.CR > 1:
            self.CR = 1
        elif self.CR < 0:
            self.CR = 0
        return self.CR

    def mutation(self):
        self.best_vecs_indexes = self.sorted_pop_indexes[:self.best_vecs_len]
        self.donor = self.current_to_pbest_A_mutation()

    def selection(self):
        if self.is_trial_better():
            if self.pop_fitness[self.ind_trgt_vec] != self.trial_fitness:
                self.SF.append(self.F)
                self.SCR.append(self.CR)
                self.A.append(self.pop[self.ind_trgt_vec])
            self.set_trial_to_target()

    def feedback(self):
        A_current_size = len(self.A)
        if A_current_size > self.AN:
            indexes = random.sample(range(A_current_size), A_current_size - self.AN)
            indexes = sorted(indexes, reverse=True)
            for index in indexes:
                del self.A[index]

        self.MF = (1 - self.C) * self.MF + self.C * Mean.lehmer(self.SF)
        self.MCR = (1 - self.C) * self.MCR + self.C * Mean.arithmetic(self.SCR)
        self.SF, self.SCR, self.trial_fitness = [], [], []


class SHADE(Adaptive):
    ALGO_NAME = "SHADE"
    ALGO_FOLDER_NAME = "SHADE"
    MUTATION_TYPE = "DE/current-to-pbest-Archive"

    def __init__(self, MF, MCR, H, AN, C=0.1, K=0, best_vecs_len_range=np.arange(0.02, 0.20+0.01, 0.01)):
        self.initialize_population()
        self.MF, self.MCR, self.H, self.C, self.K = MF, MCR, H, C, K
        self.SF, self.SCR, self.pop_trial_differences = [], [], []
        self.A = []
        self.best_vecs_len_range = best_vecs_len_range
        self.NFE = 0
        self.AN = AN

    def get_F(self):
        self.r = self.get_r()
        self.F = Rand.cauchy(loc=self.MF[self.r], scale=0.1)
        if self.F > 1:
            self.F = 1
        while self.F < 0 or self.F > 1:
            self.F = Rand.cauchy(self.MF[self.r], 0.1)
        return self.F

    def get_CR(self):
        self.CR = np.random.normal(self.MCR[self.r], 0.1)
        if self.CR > 1:
            self.CR = 1
        elif self.CR < 0:
            self.CR = 0
        return self.CR

    def get_r(self):
        return random.choice(list(range(self.H)))

    def get_best_vecs_len(self):
        return int(random.choice(self.best_vecs_len_range) * DE.PN_INIT)+1

    def mutation(self):
        self.best_vecs_indexes = self.sorted_pop_indexes[:self.get_best_vecs_len()]
        self.donor = self.current_to_pbest_A_mutation()

    def selection(self):
        if self.is_trial_better():
            if self.pop_fitness[self.ind_trgt_vec] != self.trial_fitness:
                self.A.append(self.pop[self.ind_trgt_vec])
                self.SF.append(self.F)
                self.SCR.append(self.CR)
                self.pop_trial_differences.append(abs(self.pop_fitness[self.ind_trgt_vec] - self.trial_fitness))
            self.set_trial_to_target()

    def feedback(self):
        A_current_size = len(self.A)
        if A_current_size > self.AN:
            indexes = random.sample(range(A_current_size), A_current_size - self.AN)
            indexes = sorted(indexes, reverse=True)
            for index in indexes:
                del self.A[index]

        if self.SF:
            self.MF[self.K] = Mean.lehmer_weighted(self.pop_trial_differences, self.SF)
            self.MCR[self.K] = Mean.arithmetic_weighted(self.pop_trial_differences, self.SCR)
            self.K += 1
            if self.K == self.H - 1: self.K = 0
        self.SF, self.SCR, self.trial_fitness, self.pop_trial_differences = [], [], [], []


class LSHADE(Adaptive):
    ALGO_NAME = "LSHADE"
    ALGO_FOLDER_NAME = "LSHADE"
    MUTATION_TYPE = "DE/current-to-pbest-Archive"

    def __init__(self, MF, MCR, H, AN, C=0.1, K=0, pbest=0.1):
        self.initialize_population()
        self.MF, self.MCR, self.H, self.C, self.K = MF, MCR, H, C, K
        self.SF, self.SCR, self.pop_trial_differences = [], [], []
        self.A = []
        self.pbest = pbest
        self.NFE = 0
        self.AN = AN

    def get_F(self):
        self.r = self.get_r()
        self.F = Rand.cauchy(loc=self.MF[self.r], scale=0.1)
        if self.F > 1:
            self.F = 1
        while 0 >= self.F or self.F > 1:
            self.F = Rand.cauchy(loc=self.MF[self.r], scale=0.1)
        return self.F

    def get_CR(self):
        if self.MCR[self.r] == 111:
            self.CR = 0
        else:
            self.CR = np.random.normal(self.MCR[self.r], 0.1)
            if self.CR > 1:
                self.CR = 1
            elif self.CR < 0:
                self.CR = 0
        return self.CR

    def get_r(self):
        return random.choice(list(range(self.H)))

    def get_best_vecs_len(self):
        #return int(random.choice(self.best_vecs_len_range) * self.PN) + 1
        return int(math.ceil(self.pbest * self.PN))

    def mutation(self):
        self.best_vecs_indexes = self.sorted_pop_indexes[:self.get_best_vecs_len()]
        self.donor = self.current_to_pbest_A_mutation()

    def selection(self):
        if self.is_trial_better():
            if self.pop_fitness[self.ind_trgt_vec] != self.trial_fitness:
                self.A.append(self.pop[self.ind_trgt_vec])
                self.SF.append(self.F)
                self.SCR.append(self.CR)
                self.pop_trial_differences.append(abs(self.pop_fitness[self.ind_trgt_vec] - self.trial_fitness))
            self.set_trial_to_target()

    def feedback(self):
        if self.SF and self.SCR:
            if self.MCR[self.K] == 111 or max(self.SCR) == 0.0:
                self.MCR[self.K] = 111
            else:
                self.MCR[self.K] = Mean.lehmer_weighted(self.pop_trial_differences, self.SCR)
            self.MF[self.K] = Mean.lehmer_weighted(self.pop_trial_differences, self.SF)
            self.K += 1
            if self.K == self.H - 1:
                self.K = 0

        self.SF, self.SCR, self.pop_trial_differences = [], [], []

        PN_new = round(((DE.PN_MIN - DE.PN_INIT) / DE.MAX_NFE) * self.NFE + DE.PN_INIT)
        if PN_new < self.PN:
            indexes = [self.sorted_pop_indexes[-(self.PN - PN_new)]]
            indexes = sorted(indexes, reverse=True)
            for index in indexes:
                del self.pop[index]
                del self.pop_constraints[index]
                del self.pop_fitness[index]
                del self.pop_feasibility[index]
                del self.pop_constraints_violation_num[index]
            self.PN = PN_new
            self.AN = self.PN

        A_current_size = len(self.A)
        if A_current_size > self.AN:
            indexes = random.sample(range(A_current_size), A_current_size - self.AN)
            indexes = sorted(indexes, reverse=True)
            for index in indexes:
                del self.A[index]


class Mean:
    @classmethod
    def arithmetic(cls, vec):
        try:
            return sum(vec) / len(vec)
        except ZeroDivisionError:
            return 0

    @classmethod
    def lehmer(cls, vec):
        try:
            pow_list = [math.pow(x, 2) for x in vec]
            return sum(pow_list) / sum(vec)
        except ZeroDivisionError:
            return 0

    @classmethod
    def arithmetic_weighted(cls, differences, s):
        w_denominator = sum(differences)
        w = [differences[k] / w_denominator for k in range(len(s))]
        m = [s[k] * w[k] for k in range(len(s))]
        return sum(m)

    @classmethod
    def lehmer_weighted(cls, differences, s):
        w_denominator = sum(differences)
        m_numerator, m_denominator = 0, 0
        for k in range(len(s)):
            w = differences[k] / w_denominator
            m_numerator += math.pow(s[k], 2) * w
            m_denominator += s[k] * w
        return m_numerator / m_denominator


class Rand:
    @classmethod
    def cauchy(cls, loc, scale=0.1):
        return loc + (scale * math.tan(math.pi * (random.uniform(0, 1) - 0.5)))


def DE_initialization():
    DE.static(D=Problem.D, BOUNDS=Problem.BOUNDS, GENE_NO=Problem.GENE_NUM, MAX_NFE=Problem.D * 36, PN_INIT=Problem.D * 18, PN_MIN=4)
    Tuning.static(F=0.5, CR=0.9)
    objs = [
        LSHADE(MF=[0.5] * 6, MCR=[0.5] * 6, H=6, AN=DE.PN_INIT, C=0.1, K=0, pbest=0.11),
        SHADE(MF=[0.5] * 100, MCR=[0.5] * 100, H=100, AN=DE.PN_INIT, C=0.1, K=0, best_vecs_len_range=np.arange(2/DE.PN_INIT, 0.20 + 0.01, 0.01)),
        JADEW(AN=DE.PN_INIT, pbest=0.05, MF=0.5, MCR=0.5, C=0.1),
        JADEWO(pbest=0.05, MF=0.5, MCR=0.5, C=0.1),
        jDE(Fl=0.1, Fu=0.9, T1=0.1, T2=0.1),
        DECurrentToBest1(),
        DEBest1(),
        DERand1()
    ]
    return objs


def DE_steps(obj):
    graphic_info = []
    obj.sorted_pop_indexes = obj.get_sorted_pop_indexes()

    while obj.NFE <= DE.MAX_NFE:
        for i in range(obj.PN):
            obj.ind_trgt_vec = i
            obj.mutation()
            obj.recombine()
            obj.evaluation_trial()
            obj.selection()
            obj.NFE += 1
            if obj.NFE % (obj.MAX_NFE / 20) == 0:
                obj.best_vec_index = obj.sorted_pop_indexes[0]
                graphic_info.append([obj.NFE / DE.D, obj.pop_fitness[obj.best_vec_index]])
        obj.sorted_pop_indexes = obj.get_sorted_pop_indexes()
        obj.feedback()

    obj.best_vec_index = obj.sorted_pop_indexes[0]
    print(f'Model_{Problem.MODEL_NUM} | {obj.ALGO_NAME}: Feasibility = {obj.pop_feasibility[obj.best_vec_index]} | Fitness = {obj.pop_fitness[obj.best_vec_index]}')
    data = {
        'individual': obj.pop[obj.best_vec_index],
        'graphic_info': graphic_info
    }
    return data


if __name__ == '__main__':
    DE_objects = DE_initialization()

    for obj in DE_objects:
        data = DE_steps(obj)
        Report.solution(data['individual'])
        file_name = Problem.get_file_name()
        path = 'results/' + obj.ALGO_FOLDER_NAME + '/model_' + str(Problem.MODEL_NUM)
        InputOutput.to_csv(path=path, file_name='individual_' + file_name, data=data['individual'])
        InputOutput.to_csv(path=path, file_name='graphic_info_' + file_name, data=data['graphic_info'])
