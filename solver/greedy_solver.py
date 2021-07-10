import base_solver
'''======================= Greedy solver =========================='''


class GreedySolver(base_solver.BaseSolver):
    def greedy(self, graph):
        # base cases
        if (len(graph) == 0):
            return []

        if (len(graph) == 1):
            return [list(graph.keys())[0]]

        # Greedily select a vertex from the graph
        current = list(graph.keys())[0]

        # Case 1 - current node not included
        graph2 = dict(graph)
        del graph2[current]

        res1 = self.greedy(graph2)

        # Case 2 - current node included
        # Delete its neighbors
        for v in graph[current]:
            if (v in graph2):
                del graph2[v]

        res2 = [current] + self.greedy(graph2)

        # select the maximum set
        if (len(res1) > len(res2)):
            return res1
        return res2

    def solve(self):
        edge_index = self.G.edge_index
        # print(edge_index.t())
        E = edge_index.t().tolist()

        graph = dict([])
        for i in range(len(E)):
            v1, v2 = E[i]

            if (v1 not in graph):
                graph[v1] = []
            if (v2 not in graph):
                graph[v2] = []

            if (v2 not in graph[v1]):
                graph[v1].append(v2)
            if (v1 not in graph[v2]):
                graph[v2].append(v1)

        # sort by node degree
        a = sorted(graph.items(), key=lambda x: len(x[1]))

        # Rewrite by ED
        marked = dict([])
        result = []
        for node, neighbors in a:
            if node not in marked:
                result.append(node)
                marked[node] = True
                for neighbor in neighbors:
                    marked[neighbor] = True

        # graph = dict([])
        # for i in range(len(a)):
        #     v1, v2 = a[i]
        #     graph[v1] = v2
        #
        # result = self.greedy(graph)

        self.solution = result
        self.metric()
