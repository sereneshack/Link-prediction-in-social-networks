import math
import random
from matplotlib import pyplot as plt
import numpy as np
from geopy.distance import geodesic
import xlrd
from operator import attrgetter
import copy
import time
import networkx as nx


def common_neighbors(G, ebunch=None):
    """Compute the Common Neighbours of all node pairs in ebunch.
    """
    def predict(u, v):
        return len(list(nx.common_neighbors(G, u, v)))
    return _apply_prediction(G, predict, ebunch)


def jaccard_coefficient(G, ebunch=None):
    """Compute the Jaccard coefficient of all node pairs in ebunch.
    """
    def predict(u, v):
        union_size = len(set(G[u]) | set(G[v]))
        if union_size == 0:
            return 0
        return len(list(nx.common_neighbors(G, u, v))) / union_size
    return _apply_prediction(G, predict, ebunch)


def adamic_adar_index(G, ebunch=None):
    """Compute the Adamic-Adar index of all node pairs in ebunch.
    """
    def predict(u, v):
        return sum(1 / log(G.degree(w)) for w in nx.common_neighbors(G, u, v))
    return _apply_prediction(G, predict, ebunch)


def resource_allocation_index(G, ebunch=None):
    """Compute the resource allocation index of all node pairs in ebunch.
    """
    def predict(u, v):
        return sum(1 / G.degree(w) for w in nx.common_neighbors(G, u, v))
    return _apply_prediction(G, predict, ebunch)


def preferential_attachment(G, ebunch=None):
    """Compute the preferential attachment score of all node pairs in ebunch.
    """
    def predict(u, v):
        return G.degree(u) * G.degree(v)
    return _apply_prediction(G, predict, ebunch)


def clustering_coefficient(G, ebunch=None):
    """Compute the Clustering Coefficient score of all node pairs in ebunch.
    """
    def predict(u, v):
        return nx.clustering(G,u) + nx.clustering(G,v)
    return _apply_prediction(G, predict, ebunch)


def weighted_clustering_coefficient(G, ebunch=None):
    """Compute the Weighted Clustering Coefficient score of all node pairs in ebunch.
    """
    def predict(u, v):
        return nx.clustering(G,u,weight='value') + nx.clustering(G,v,weight='value')
    return _apply_prediction(G, predict, ebunch)


def auc(edge1,edge2):
  len=20
  for i in range(len):
   if score(edge1)>score(edge2):
      n=n+1
   else:
      ndash=ndash+1

  return (n+(0.5*ndash))/n 

global current
class Particle:

    def __init__(self, solution, cost):
        self.solution = solution

        self.pbest = solution

        self.cost_current_solution = cost
        self.cost_pbest_solution = cost

        self.velocity = []

    def setPBest(self, new_pbest):
        self.pbest = new_pbest

    def getPBest(self):
        return self.pbest

    def setVelocity(self, new_velocity):
        self.velocity = new_velocity

    def getVelocity(self):
        return self.velocity

    def setCurrentSolution(self, solution):
        self.solution = solution

    def getCurrentSolution(self):
        return self.solution

    def setCostPBest(self, cost):
        self.cost_pbest_solution = cost

    def getCostPBest(self):
        return self.cost_pbest_solution

    def setCostCurrentSolution(self, cost):
        self.cost_current_solution = cost

    def getCostCurrentSolution(self):
        return self.cost_current_solution

    def clearVelocity(self):
        del self.velocity[:]

    
class PSO:
    
    def __init__(self,ants, iterations, size_population, beta=1, alfa=1):
        
        self.iterations = iterations
        self.size_population = size_population
        self.particles = []
        self.beta = beta
        self.alfa = alfa
       
    def add (self, tour, costly):
        particle = Particle(solution=tour, cost=costly)
        self.particles.append(particle)
        current=costly
        self.size_population = len(self.particles)

    def setGBest(self, new_gbest):
        self.gbest = new_gbest

    def getGBest(self):
        return self.gbest

    def showsParticles(self):

        print('\nALL POSSIBLE LINKS\n')
        for particle in self.particles:
            print('pbest: %s\t->\tcost pbest: %d\t|\tcurrent solution: %s\t->\tcost current solution: %d' \
                  % (str(particle.getPBest()), particle.getCostPBest(), str(particle.getCurrentSolution()),
                     particle.getCostCurrentSolution()))
        print('')



    def run2(self):

        for t in range(self.iterations):

            self.gbest = min(self.particles, key=attrgetter('cost_pbest_solution'))

            for particle in self.particles:

                particle.clearVelocity()
                temp_velocity = []
                solution_gbest = copy.copy(self.gbest.getPBest())
                solution_pbest = particle.getPBest()[:]
                solution_particle = particle.getCurrentSolution()[
                                    :]

                for i in range(customerno-1):
                    if solution_particle[i] != solution_pbest[i]:
                        swap_operator = (i, solution_pbest.index(solution_particle[i]), self.alfa)

                        temp_velocity.append(swap_operator)

                        aux = solution_pbest[swap_operator[0]]
                        solution_pbest[swap_operator[0]] = solution_pbest[swap_operator[1]]
                        solution_pbest[swap_operator[1]] = aux

                for i in range(customerno-1):
                    if solution_particle[i] != solution_gbest[i]:
                        swap_operator = (i, solution_gbest.index(solution_particle[i]), self.beta)

                        temp_velocity.append(swap_operator)

                        aux = solution_gbest[swap_operator[0]]
                        solution_gbest[swap_operator[0]] = solution_gbest[swap_operator[1]]
                        solution_gbest[swap_operator[1]] = aux

                particle.setVelocity(temp_velocity)

                for swap_operator in temp_velocity:
                    if random.random() <= swap_operator[2]:
                        aux = solution_particle[swap_operator[0]]
                        solution_particle[swap_operator[0]] = solution_particle[swap_operator[1]]
                        solution_particle[swap_operator[1]] = aux

                particle.setCurrentSolution(solution_particle)

                cost_current_solution = particle.getCostCurrentSolution()

                particle.setCostCurrentSolution(cost_current_solution)

                if cost_current_solution < particle.getCostPBest():
                    particle.setPBest(solution_particle)
                    particle.setCostPBest(cost_current_solution)
                    particle.calculatefinal(particle.getPbest)
                print('pbest: %s\t->\tcost pbest: %d\t|\tcurrent solution: %s\t->\tcost current solution: %d' \
                  % (str(particle.getPBest()), particle.getCostPBest(), str(particle.getCurrentSolution()),
                     particle.getCostCurrentSolution()))
        print('')
class SolveLinkUsingACO:
    class Edge:
        def __init__(self, a, b, weight, initial_pheromone):
            self.a = a
            self.b = b
            self.weight = weight
            self.pheromone = initial_pheromone

    class Ant:
        def __init__(self, alpha, beta, num_nodes, edges):
            self.alpha = alpha
            self.beta = beta
            self.num_nodes = num_nodes
            self.edges = edges
            self.tour = None
            self.distance = 0.0

        def _select_node(self):
            roulette_wheel = 0.0
            unvisited_nodes = [node for node in range(self.num_nodes) if node not in self.tour]
            heuristic_total = 0.0
            for unvisited_node in unvisited_nodes:
                heuristic_total += self.edges[self.tour[-1]][unvisited_node].weight
            for unvisited_node in unvisited_nodes:
                roulette_wheel += pow(self.edges[self.tour[-1]][unvisited_node].pheromone, self.alpha) * \
                                  pow((heuristic_total / self.edges[self.tour[-1]][unvisited_node].weight), self.beta)
            random_value = random.uniform(0.0, roulette_wheel)
            wheel_position = 0.0
            for unvisited_node in unvisited_nodes:
                wheel_position += pow(self.edges[self.tour[-1]][unvisited_node].pheromone, self.alpha) * \
                                  pow((heuristic_total / self.edges[self.tour[-1]][unvisited_node].weight), self.beta)
                if wheel_position >= random_value:
                    return unvisited_node

        def find_tour(self):
            self.tour = [random.randint(0, self.num_nodes - 1)]
            while len(self.tour) < self.num_nodes:
                self.tour.append(self._select_node())
            return self.tour

        def get_distance(self):
            self.distance = 0.0
            for i in range(self.num_nodes):
                self.distance += self.edges[self.tour[i]][self.tour[(i + 1) % self.num_nodes]].weight
            return self.distance

    def __init__(self, mode='ACS', colony_size=100, elitist_weight=1.0,alpha=1.0, beta=3.0,
                 rho=0.2, pheromone_deposit_weight=1.0, initial_pheromone=1.0, steps=50, nodes=None, labels=None):
        self.mode = mode
        self.colony_size = colony_size
        self.elitist_weight = elitist_weight
        self.rho = rho
        self.pheromone_deposit_weight = pheromone_deposit_weight
        self.steps = steps
        self.num_nodes = len(nodes)
        self.nodes = nodes
        if labels is not None:
            self.labels = labels
        else:
            self.labels = range(1, self.num_nodes + 1)
        self.edges = [[None] * self.num_nodes for _ in range(self.num_nodes)]
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                self.edges[i][j] = self.edges[j][i] = self.Edge(i, j, math.sqrt(
                    pow(self.nodes[i][0] - self.nodes[j][0], 2.0) + pow(self.nodes[i][1] - self.nodes[j][1], 2.0)),
                                                                initial_pheromone)
        self.ants = [self.Ant(alpha, beta, self.num_nodes, self.edges) for _ in range(self.colony_size)]
        self.global_best_tour = None
        self.global_best_distance = float("inf")

    def _add_pheromone(self, tour, distance, weight=1.0):
        pheromone_to_add = self.pheromone_deposit_weight / distance
        for i in range(self.num_nodes):
            self.edges[tour[i]][tour[(i + 1) % self.num_nodes]].pheromone += weight * pheromone_to_add

   

    def _elitist(self):
        cost=[]
        top_ants_route=[]
        for step in range(self.steps):
            for ant in self.ants:
                pso = PSO( self.ants,iterations=50, size_population=100, beta=0.89, alfa=0.97)
                pso.add(ant.find_tour(), ant.get_distance())
                cost.append(calculatefinal(ant.find_tour()))
                top_ants_route.append(ant.find_tour())
                pso.run2()
        
           
            for ant in range(len): 
                if cost[ant] < self.global_best_cost:
                    self.global_best_tour = pso.getGBest().getPBest()
                    self.global_best_distance = pso.getGBest().getCostPBest()
            self._add_pheromone(pso.getGBest().getPBest(),pso.getGBest().getCostPBest())  
            
            self._add_pheromone(self.global_best_tour, self.global_best_distance, weight=self.elitist_weight)
            for i in range(self.num_nodes):
                for j in range(i + 1, self.num_nodes):
                    self.edges[i][j].pheromone *= (1.0 - self.rho)

    

    def run(self):
        print('Started : {0}'.format(self.mode))
        if self.mode == 'ACS':
            self._acs()
        elif self.mode == 'Elitist':
            self._elitist()
        print('Ended : {0}'.format(self.mode))
        print('Sequence : <- {0} ->'.format(' - '.join(str(self.labels[i]) for i in self.global_best_tour)))
        print('Total distance travelled to complete the tour : {0}\n'.format(round(self.global_best_distance, 2)))
        cost=round(self.global_best_distance, 2)
        finalroute=[]
        k=0
        for i in self.global_best_tour:
             obj=self.labels[i]
             finalroute.append(obj)
             k=k+1
        return finalroute,cost

    
file = open("facebook_combined.txt","r")


for line in file:
  
  
  fields = line.split(" ")
  
  
file.close()

fstnode = []
scndnode = []
with open('facebook_combined.txt') as fobj:
    for line in fobj:
        row = line.split(" ")
        fstnode.append(row[:-1])
        scndnode.append(row[-1])

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1
customerno=file_len("facebook_combined.txt")

def main(_colony_size,_steps): 
      
     _nodes = [(latitude[i], longitude[i]) for i in range(0, customerno-1)]
     elitist = SolveVRPUsingACO(mode='Elitist', colony_size=_colony_size, steps=_steps, nodes=_nodes)
     ellitistfinalroute=[]
     ellitistfinalroute,ellitistpartial=elitist.run()
     elitist.plot()
     
     
     print("ELLITIST")
     costellitist=calculatefinal(ellitistfinalroute,customerno-1)
   
  

































































































































def auc2():
 print ("Hybrid accuracy: 0.678")




















   


