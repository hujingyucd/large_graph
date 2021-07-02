import base_solver
import random

'''======================= Random solver =========================='''
class RandomSolver(base_solver.BaseSolver):
    def random(self, graph):
        # Base cases
        if(len(graph) == 0):
            return []

        if(len(graph) == 1):
            return [list(graph.keys())[0]]
          
        # Randomly select a vertex from the graph
        vCurrent = random.choice(list(graph.keys()))
          
        # Case 1 - current node not included
        graph2 = dict(graph)
        del graph2[vCurrent]
          
        res1 = self.random(graph2)
          
        # Case 2 - current node included
        # Delete its neighbors
        for v in graph[vCurrent]:
            if(v in graph2):
                del graph2[v]
          
        res2 = [vCurrent] + self.random(graph2)
          
        # select the maximum set
        if(len(res1) > len(res2)):
            return res1
        return res2
        
    def solve(self):
        edge_index = self.G.edge_index
        E = edge_index.t().tolist()
        
        graph = dict([])
        for i in range(len(E)):
            v1, v2 = E[i]
              
            if(v1 not in graph):
                graph[v1] = []
            if(v2 not in graph):
                graph[v2] = []
            
            if(v2 not in graph[v1]):
                graph[v1].append(v2)
            if(v1 not in graph[v2]):
                graph[v2].append(v1)
        
        result = self.random(graph)
        self.solution = result
        self.metric()
