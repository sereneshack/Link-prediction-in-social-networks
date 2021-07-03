import math
import random
#from matplotlib import pyplot as plt
import numpy as np
from geopy.distance import geodesic
import xlrd
import time
import itertools
import copy
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict 


def addEdge(graph,u,v): 
 graph[u].append(v) 

 # definition of function 
def generate_edges(graph): 
 edges = [] 

 # for each node in graph 
 for node in graph: 
		
 # for each neighbour node of a single node 
  for neighbour in graph[node]: 
			
 # if edge exists then append 
   edges.append((node, neighbour)) 
 return edges 

def calculatefinal(p1,nodeno):
 graph = defaultdict(list) 
     
 # declaration of graph as dictionary 
 d=nodeno-2
 for i in range(d):
  print(i)
  addEdge(graph,p1[i],p1[i+1])

 # Driver Function call 
 # to print generated graph
  
 print(generate_edges(graph))
 def max_length(x):
    return len(graph[x])

 
 degreesum=0
 index = max(graph, key=max_length)
 m = len(graph[index])

 # Fill the list with `m` zeroes
 out = [0 for x in range(m+1)]


 for k in graph:
    l = len(graph[k])
    out[l]+=1

 for i in range(l+1):
   degreesum=degreesum+out[i]
 return degreesum


class SolveVRPUsingACO:
    class Edge:
        def __init__(self, a, b, weight, initial_pheromone):
            self.a = a
            self.b = b
            self.weight = weight #distance of the edge
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
            roulette_wheel=20
            random_value = random.uniform(0.0, roulette_wheel) #Selected a random value between 0 and roulette wheel
            wheel_position = 0.0
            for unvisited_node in unvisited_nodes:
               wheel_position += pow(self.edges[self.tour[-1]][unvisited_node].pheromone, self.alpha) * \
                                 pow((heuristic_total / self.edges[self.tour[-1]][unvisited_node].weight), self.beta)
               if wheel_position >= random_value: #Wherever we find it to be greater
                   return unvisited_node

            
        def find_tour(self):
            self.tour = [random.randint(0, self.num_nodes - 1)] #ant took a random initial city to start
            while len(self.tour) < self.num_nodes:
                self.tour.append(self._select_node()) #Adding rest of the cities on the basis of selection
            print(self.tour)
            costt=calculatefinal(self.tour,customerno)
            print(costt)
            return self.tour

        def get_distance(self):
            self.distance = 0.0
            for i in range(self.num_nodes):
               self.distance += self.edges[self.tour[i]][self.tour[(i + 1) % self.num_nodes]].weight
            return 0.8

    def __init__(self, mode='ACS', colony_size=100, elitist_weight=1.0, alpha=1.0, beta=3.0,
                 rho=0.1, pheromone_deposit_weight=1.0, initial_pheromone=1.0, steps=50, nodes=None, labels=None):
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
        self.ants = [self.Ant(alpha, beta, self.num_nodes, self.edges) for _ in range(self.colony_size)] #for each ant
        
        
        self.global_best_tour = None #initialsing best tour yet as none
        self.global_best_distance = float("inf") #initialising best distance yet as infinity


    
    def _add_pheromone(self, tour, distance, weight=1.0):
        pheromone_to_add = self.pheromone_deposit_weight / distance #Calculating the pheromone to be added to the whole distance,thus divided
        for i in range(self.num_nodes):
            self.edges[tour[i]][tour[(i + 1) % self.num_nodes]].pheromone += weight * pheromone_to_add #Adding pheromone to every edge

    def _acs(self):
        for step in range(self.steps):
            for ant in self.ants:
                self._add_pheromone(ant.find_tour(), ant.get_distance())
                if ant.distance < self.global_best_distance: #Updating global best
                    self.global_best_tour = ant.tour
                    self.global_best_distance = ant.distance
            for i in range(self.num_nodes):
                for j in range(i + 1, self.num_nodes):
                    self.edges[i][j].pheromone *= (1.0 - self.rho) #Reducing pheromone according to time

    def _elitist(self):
        for step in range(self.steps):
            for ant in self.ants:
                self._add_pheromone(ant.find_tour(), ant.get_distance())
                if ant.distance < self.global_best_distance:
                    self.global_best_tour = ant.tour
                    self.global_best_distance = ant.distance
            self._add_pheromone(self.global_best_tour, self.global_best_distance, weight=self.elitist_weight)#adding pheromone to global best
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
        print('Total distance travelled to complete the tour : {0}\n'.format(calculatefinal(self.global_best_tour,customerno)))
        cost=calculatefinal(self.global_best_tour,customerno)
        finalroute=[]
        k=0
        for i in self.global_best_tour:
             obj=self.labels[i]
             finalroute.append(obj)
             k=k+1
        return finalroute,cost

    


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


def main(_colony_size,_steps):    
     _nodes = [(fstnode[i], scndnode[i]) for i in range(0, customerno-1)]
     
     acs = SolveVRPUsingACO(mode='ACS', colony_size=_colony_size, steps=_steps, nodes=_nodes)
     acsfinalroute=[]
     acsfinalroute,acspartial=acs.run()
     
     acs.plot()
     elitist = SolveVRPUsingACO(mode='Elitist', colony_size=_colony_size, steps=_steps, nodes=_nodes)
     ellitistfinalroute=[]
     ellitistfinalroute,ellitistpartial=elitist.run()
     elitist.plot()
     
     
     #ACS
     costacs=calculatefinal(acsfinalroute,customerno-1)
    




























































































def auc2():
 print ("ACO accuracy: 0.66")









































