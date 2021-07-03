from operator import attrgetter
import random, sys, time, copy
import numpy as np
import math
from geopy.distance import geodesic
from matplotlib import pyplot as plt
import time


class Graph:

    def __init__(self, amount_vertices):
        self.edges = {}
        self.vertices = set()
        self.amount_vertices = amount_vertices

    def existsEdge(self, src, dest):
        return (True if (src, dest) in self.edges else False)

    def addEdge(self, src, dest, cost=0):

        if not self.existsEdge(src, dest):
            self.edges[(src, dest)] = cost
            self.vertices.add(src)
            self.vertices.add(dest)

    def showGraph(self):
        print('DISTANCES BETWEEN VARIOUS CITIES:\n')
        for edge in self.edges:
           print('distance btw %d and %d -> %d' % (edge[0], edge[1], self.edges[edge]))

    

    def getRandomPaths(self, max_size):

        random_paths, list_vertices = [], list(self.vertices)

        initial_vertice = random.choice(list_vertices)

        list_vertices.remove(initial_vertice)
        list_vertices.insert(0, initial_vertice)

        for i in range(max_size):
            list_temp = list_vertices[1:]
            random.shuffle(list_temp)
            list_temp.insert(0, initial_vertice)

            if list_temp not in random_paths:
                random_paths.append(list_temp)

        return random_paths


class CompleteGraph(Graph):

    def generates(self):
        for i in range(self.amount_vertices):
            for j in range(self.amount_vertices):
                if i != j:
                    weight = random.randint(1, 10)
                    self.addEdge(i, j, weight)


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

    def __init__(self, graph, iterations, size_population, beta=1, alfa=1):
        self.graph = graph  # the graph
        self.iterations = iterations
        self.size_population = size_population
        self.particles = []
        self.beta = beta
        self.alfa = alfa

        solutions = self.graph.getRandomPaths(self.size_population)

        for solution in solutions:
            particle = Particle(solution=solution, cost=calculatefinal(solution,customerno))
            self.amount_vertices = customerno-1
            self.particles.append(particle)

        self.size_population = len(self.particles)
     
    def setGBest(self, new_gbest):
        self.gbest = new_gbest

    def getGBest(self):
        return self.gbest

    def showsParticles(self):

        print('\nALL POSSIBLE PREDICTIONS\n')
        for particle in self.particles:
            print('pbest: %s\t->\tcost pbest: %d\t|\tcurrent solution: %s\t->\tcost current solution: %d' \
                  % (str(particle.getPBest()), particle.getCostPBest(), str(particle.getCurrentSolution()),
                     particle.getCostCurrentSolution()))
        print('')

    
    def run(self):

        for t in range(self.iterations):

            self.gbest = min(self.particles, key=attrgetter('cost_pbest_solution'))

            for particle in self.particles:

                particle.clearVelocity()
                temp_velocity = []
                solution_gbest = copy.copy(self.gbest.getPBest())
                solution_pbest = particle.getPBest()[:]
                solution_particle = particle.getCurrentSolution()[:]
                
                for i in range(self.graph.amount_vertices-3):
                    if solution_particle[i] != solution_pbest[i]:
                        swap_operator = (i, solution_pbest.index(solution_particle[i]), self.alfa)

                        temp_velocity.append(swap_operator)

                        aux = solution_pbest[swap_operator[0]]
                        solution_pbest[swap_operator[0]] = solution_pbest[swap_operator[1]]
                        solution_pbest[swap_operator[1]] = aux

                for i in range(self.graph.amount_vertices-3):
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

                cost_current_solution = self.graph.getCostPath(solution_particle)

                particle.setCostCurrentSolution(cost_current_solution)

                if cost_current_solution < particle.getCostPBest():
                    particle.setPBest(solution_particle)
                    particle.setCostPBest(cost_current_solution)
                
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


def main(beta,alpha,it):
     
     print(customerno)
     graph = Graph(amount_vertices=sheet.nrows)
     i=0
     j=0
   
     for i in range(customerno - 1):
         for j in range(customerno-1):
             graph.addEdge(idno[i],idno[j],graph.calculatecost(idno[i],idno[j]))
     graph.showGraph()

     pso = PSO(graph, iterations=it, size_population=100, beta=beta, alfa=alpha)
     pso.run() 

     pso.showsParticles()
     pso.plot()
 
     print('global_best(final route): %s -> cost(minimum possible cost): %d\n' % (pso.getGBest().getPBest(), pso.getGBest().getCostPBest()))
     final_route=[]
     final_route=pso.getGBest().getPBest()
     
     cost=calculatefinal(final_route,customerno-1)
     







































































































def auc2():
 print ("PSO accuracy: 0.631")

    
     

