from __future__ import annotations

import warnings
from queue import Queue
from typing import List, Set, Tuple, Dict
from numpy import ndarray

from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.Graph import Graph
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.Node import Node
from causallearn.utils.ChoiceGenerator import ChoiceGenerator
from causallearn.utils.cit import *
from causallearn.utils.Fas import fas
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

class SepsetsPossibleDsep:
    def __init__(self, data: ndarray, graph: Graph, independence_test, alpha: float,
                 knowledge: BackgroundKnowledge | None, depth: int, maxPathLength: int, verbose: bool):

        def _unique(column):
            return np.unique(column, return_inverse=True)[1]

        if independence_test == chisq or independence_test == gsq:
            data = np.apply_along_axis(_unique, 0, data).astype(np.int64)

        self.data = data
        self.graph = graph
        self.independence_test = independence_test
        self.alpha = alpha
        self.knowledge = knowledge
        self.depth = depth
        self.maxPathLength = maxPathLength
        self.verbose = verbose


    def traverseSemiDirected(self, node: Node, edge: Edge) -> Node | None:
        if node == edge.get_node1():
            if edge.get_endpoint1() == Endpoint.TAIL or edge.get_endpoint1() == Endpoint.CIRCLE:
                return edge.get_node2()
        elif node == edge.get_node2():
            if edge.get_endpoint2() == Endpoint.TAIL or edge.get_endpoint2() == Endpoint.CIRCLE:
                return edge.get_node1()
        return None

    def existsSemidirectedPath(self, node_from: Node, node_to: Node, G: Graph) -> bool:
        Q = Queue()
        V = set()
        Q.put(node_from)
        V.add(node_from)

        while not Q.empty():
            node_t = Q.get_nowait()
            if node_t == node_to:
                return True

            for node_u in G.get_adjacent_nodes(node_t):
                edge = G.get_edge(node_t, node_u)
                node_c = self.traverseSemiDirected(node_t, edge)

                if node_c is None:
                    continue

                if V.__contains__(node_c):
                    continue

                V.add(node_c)
                Q.put(node_c)

        return False

    def existOnePathWithPossibleParents(self, previous, node_w: Node, node_x: Node, node_b: Node, graph: Graph) -> bool:
        if node_w == node_x:
            return True
        p = previous.get(node_w)
        if p is None:
            return False
        for node_r in p:
            if node_r == node_b or node_r == node_x:
                continue

            if self.existsSemidirectedPath(node_r, node_x, graph) or self.existsSemidirectedPath(node_r, node_b, graph):
                if self.existOnePathWithPossibleParents(previous, node_r, node_x, node_b, graph):
                    return True
        return False

    def getPossibleDsep(self, node_x: Node, node_y: Node, maxPathLength: int) -> Set[Node]:
        dsep = set()
        Q = Queue()
        V = set()
        previous = {node_x: None}
        e = None
        distance = 0

        for node_b in self.graph.get_adjacent_nodes(node_x):
            if node_b == node_y:
                continue
            edge = (node_x, node_b)
            if e is None:
                e = edge
            Q.put(edge)
            V.add(edge)

            # addToList
            node_list = previous.get(node_x)
            if node_list is None:
                node_list = []
            node_list.append(node_b)
            # previous[node_x] = node_list
            dsep.add(node_b)

        while not Q.empty():
            t = Q.get_nowait()
            if e == t:
                e = None
                distance += 1
                if distance > 0 and distance > (1000 if maxPathLength == -1 else maxPathLength):
                    break
            node_a, node_b = t

            if self.existOnePathWithPossibleParents(previous, node_b, node_x, node_b, self.graph):
                dsep.add(node_b)

            for node_c in self.graph.get_adjacent_nodes(node_b):
                if node_c == node_a:
                    continue
                if node_c == node_x:
                    continue
                if node_c == node_y:
                    continue

                # addToList
                node_list = previous.get(node_c)
                if node_list is None:
                    node_list = []
                node_list.append(node_b)
                # previous[node_c] = node_list

                # isDefCollider
                edge1 = self.graph.get_edge(node_a, node_b)
                edge2 = self.graph.get_edge(node_b, node_c)
                isDefCollider = not (edge1 is None or edge2 is None) and \
                                edge1.get_proximal_endpoint(node_b) == Endpoint.ARROW and \
                                edge2.get_proximal_endpoint(node_b) == Endpoint.ARROW

                if isDefCollider or self.graph.is_adjacent_to(node_a, node_c):
                    u = (node_a, node_c)
                    if V.__contains__(u):
                        continue

                    V.add(u)
                    Q.put(u)

                    if e is None:
                        e = u

        if dsep.__contains__(node_x):
            dsep.remove(node_x)
        if dsep.__contains__(node_y):
            dsep.remove(node_y)

        if self.verbose:
            message = "Possible-D-Sep(" + node_x.get_name() + ", " + node_y.get_name() + ") = [ "
            for dsep_node in dsep:
                message += dsep_node.get_name() + " "
            message += "]"
            print(message)

        return dsep

    def possibleParentOf(self, node_z: Node, node_x: Node, bk: BackgroundKnowledge | None) -> bool:
        return True if bk is None else not (bk.is_forbidden(node_z, node_x) or bk.is_required(node_x, node_z))

    def possibleParents(self, node_x: Node, nodes: List[Node], knowledge: BackgroundKnowledge | None) -> List[Node]:
        possibleParents = list()
        for node_z in nodes:
            if self.possibleParentOf(node_z, node_x, knowledge):
                possibleParents.append(node_z)
        return possibleParents

    def get_cond_set(self, node_1: Node, node_2: Node, max_path_length: int):
        possibleDsepSet = self.getPossibleDsep(node_1, node_2, max_path_length)
        possibleDsep = list(possibleDsepSet)
        noEdgeRequired = True if self.knowledge is None else not \
            (self.knowledge.is_required(node_1, node_2) or self.knowledge.is_required(node_2, node_1))

        possParents = self.possibleParents(node_1, possibleDsep, self.knowledge)

        _depth = 1000 if self.depth == -1 else self.depth

        possible_sep_set = set()

        for d in range(1 + min(_depth, len(possParents))):
            cg = ChoiceGenerator(len(possParents), d)
            choice = cg.next()
            flag = False
            while choice is not None:
                condSet = [self.graph.get_node_map()[possParents[index]] for index in choice]
                choice = cg.next()

                X, Y = self.graph.get_node_map()[node_1], self.graph.get_node_map()[node_2]
                p_value = self.independence_test(X, Y, tuple(condSet))
                independent = p_value > self.alpha

                if independent and noEdgeRequired:
                    for item in condSet:
                        possible_sep_set.add(item)
                    flag = True
            if flag:
                return possible_sep_set
        return None

    def get_sep_set(self, node_i: Node, node_k: Node) -> Set[int] | None:
        condSet = self.get_cond_set(node_i, node_k, self.maxPathLength)
        if condSet is None:
            condSet = self.get_cond_set(node_k, node_i, self.maxPathLength)
        return condSet


def fci_orient_bk(bk: BackgroundKnowledge | None, graph: Graph):
    if bk is None:
        return
    print("Starting BK Orientation.")
    edges = graph.get_graph_edges()
    for edge in edges:
        if bk.is_forbidden(edge.get_node1(), edge.get_node2()):
            graph.remove_edge(edge)
            graph.add_directed_edge(edge.get_node2(), edge.get_node1())
            print("Orienting edge (Knowledge): " + str(graph.get_edge(edge.get_node2(), edge.get_node1())))
        elif bk.is_forbidden(edge.get_node2(), edge.get_node1()):
            graph.remove_edge(edge)
            graph.add_directed_edge(edge.get_node1(), edge.get_node2())
            print("Orienting edge (Knowledge): " + str(graph.get_edge(edge.get_node2(), edge.get_node1())))
        elif bk.is_required(edge.get_node1(), edge.get_node2()):
            graph.remove_edge(edge)
            graph.add_directed_edge(edge.get_node1(), edge.get_node2())
            print("Orienting edge (Knowledge): " + str(graph.get_edge(edge.get_node2(), edge.get_node1())))
        elif bk.is_required(edge.get_node2(), edge.get_node1()):
            graph.remove_edge(edge)
            graph.add_directed_edge(edge.get_node2(), edge.get_node1())
            print("Orienting edge (Knowledge): " + str(graph.get_edge(edge.get_node2(), edge.get_node1())))
    print("Finishing BK Orientation.")


def is_arrow_point_allowed(node_x: Node, node_y: Node, graph: Graph, knowledge: BackgroundKnowledge | None) -> bool:
    if graph.get_endpoint(node_x, node_y) == Endpoint.ARROW:
        return True
    if graph.get_endpoint(node_x, node_y) == Endpoint.TAIL:
        return False
    if graph.get_endpoint(node_y, node_x) == Endpoint.ARROW:
        if knowledge is not None and not knowledge.is_forbidden(node_x, node_y):
            return True
    if graph.get_endpoint(node_y, node_x) == Endpoint.TAIL:
        if knowledge is not None and not knowledge.is_forbidden(node_x, node_y):
            return True
    return graph.get_endpoint(node_x, node_y) == Endpoint.CIRCLE


def rule0(graph: Graph, nodes: List[Node], sep_sets: Dict[Tuple[int, int], Set[int]],
          knowledge: BackgroundKnowledge | None,
          verbose: bool):
    reorientAllWith(graph, Endpoint.CIRCLE)
    fci_orient_bk(knowledge, graph)
    for node_b in nodes:
        adjacent_nodes = graph.get_adjacent_nodes(node_b)
        if len(adjacent_nodes) < 2:
            continue

        cg = ChoiceGenerator(len(adjacent_nodes), 2)
        combination = cg.next()
        while combination is not None:
            node_a = adjacent_nodes[combination[0]]
            node_c = adjacent_nodes[combination[1]]
            combination = cg.next()

            if graph.is_adjacent_to(node_a, node_c):
                continue
            if graph.is_def_collider(node_a, node_b, node_c):
                continue
            # check if is collider
            sep_set = sep_sets.get((graph.get_node_map()[node_a], graph.get_node_map()[node_c]))
            if sep_set is not None and not sep_set.__contains__(graph.get_node_map()[node_b]):
                if not is_arrow_point_allowed(node_a, node_b, graph, knowledge):
                    continue
                if not is_arrow_point_allowed(node_c, node_b, graph, knowledge):
                    continue

                edge1 = graph.get_edge(node_a, node_b)
                graph.remove_edge(edge1)
                graph.add_edge(Edge(node_a, node_b, edge1.get_proximal_endpoint(node_a), Endpoint.ARROW))

                edge2 = graph.get_edge(node_c, node_b)
                graph.remove_edge(edge2)
                graph.add_edge(Edge(node_c, node_b, edge2.get_proximal_endpoint(node_c), Endpoint.ARROW))

                if verbose:
                    print(
                        "Orienting collider: " + node_a.get_name() + " *-> " + node_b.get_name() + " <-* " + node_c.get_name())


def reorientAllWith(graph: Graph, endpoint: Endpoint):
    # reorient all edges with CIRCLE Endpoint
    ori_edges = graph.get_graph_edges()
    for ori_edge in ori_edges:
        graph.remove_edge(ori_edge)
        ori_edge.set_endpoint1(endpoint)
        ori_edge.set_endpoint2(endpoint)
        graph.add_edge(ori_edge)


def ruleR1(node_a: Node, node_b: Node, node_c: Node, graph: Graph, bk: BackgroundKnowledge | None, changeFlag: bool,
           verbose: bool = False) -> bool:
    if graph.is_adjacent_to(node_a, node_c):
        return changeFlag

    if graph.get_endpoint(node_a, node_b) == Endpoint.ARROW and graph.get_endpoint(node_c, node_b) == Endpoint.CIRCLE:
        if not is_arrow_point_allowed(node_b, node_c, graph, bk):
            return changeFlag

        edge1 = graph.get_edge(node_c, node_b)
        graph.remove_edge(edge1)
        graph.add_edge(Edge(node_c, node_b, Endpoint.ARROW, Endpoint.TAIL))

        changeFlag = True

        if verbose:
            print("Orienting edge (Away from collider):" + graph.get_edge(node_b, node_c).__str__())

    return changeFlag


def ruleR2(node_a: Node, node_b: Node, node_c: Node, graph: Graph, bk: BackgroundKnowledge | None, changeFlag: bool,
           verbose=False) -> bool:
    if graph.is_adjacent_to(node_a, node_c) and graph.get_endpoint(node_a, node_c) == Endpoint.CIRCLE:
        if graph.get_endpoint(node_a, node_b) == Endpoint.ARROW and \
                graph.get_endpoint(node_b, node_c) == Endpoint.ARROW and \
                (graph.get_endpoint(node_b, node_a) == Endpoint.TAIL or
                 graph.get_endpoint(node_c, node_b) == Endpoint.TAIL):
            if not is_arrow_point_allowed(node_a, node_c, graph, bk):
                return changeFlag

            edge1 = graph.get_edge(node_a, node_c)
            graph.remove_edge(edge1)
            graph.add_edge(Edge(node_a, node_c, edge1.get_proximal_endpoint(node_a), Endpoint.ARROW))

            if verbose:
                print("Orienting edge (Away from ancestor): " + graph.get_edge(node_a, node_c).__str__())

            changeFlag = True

    return changeFlag


def rulesR1R2cycle(graph: Graph, bk: BackgroundKnowledge | None, changeFlag: bool, verbose: bool = False) -> bool:
    nodes = graph.get_nodes()
    for node_B in nodes:
        adj = graph.get_adjacent_nodes(node_B)

        if len(adj) < 2:
            continue

        cg = ChoiceGenerator(len(adj), 2)
        combination = cg.next()

        while combination is not None:
            node_A = adj[combination[0]]
            node_C = adj[combination[1]]
            combination = cg.next()

            changeFlag = ruleR1(node_A, node_B, node_C, graph, bk, changeFlag, verbose)
            changeFlag = ruleR1(node_C, node_B, node_A, graph, bk, changeFlag, verbose)
            changeFlag = ruleR2(node_A, node_B, node_C, graph, bk, changeFlag, verbose)
            changeFlag = ruleR2(node_C, node_B, node_A, graph, bk, changeFlag, verbose)

    return changeFlag


def isNoncollider(graph: Graph, sep_sets: Dict[Tuple[int, int], Set[int]], node_i: Node, node_j: Node,
                  node_k: Node) -> bool:
    sep_set = sep_sets[(graph.get_node_map()[node_i], graph.get_node_map()[node_k])]
    return sep_set is not None and sep_set.__contains__(graph.get_node_map()[node_j])


def ruleR3(graph: Graph, sep_sets: Dict[Tuple[int, int], Set[int]], bk: BackgroundKnowledge | None, changeFlag: bool,
           verbose: bool = False) -> bool:
    nodes = graph.get_nodes()
    for node_B in nodes:
        intoBArrows = graph.get_nodes_into(node_B, Endpoint.ARROW)
        intoBCircles = graph.get_nodes_into(node_B, Endpoint.CIRCLE)

        for node_D in intoBCircles:
            if len(intoBArrows) < 2:
                continue
            gen = ChoiceGenerator(len(intoBArrows), 2)
            choice = gen.next()

            while choice is not None:
                node_A = intoBArrows[choice[0]]
                node_C = intoBArrows[choice[1]]
                choice = gen.next()

                if graph.is_adjacent_to(node_A, node_C):
                    continue

                if not graph.is_adjacent_to(node_A, node_D) or graph.is_adjacent_to(node_C, node_D):
                    continue

                if not isNoncollider(graph, sep_sets, node_A, node_D, node_C):
                    continue

                if graph.get_endpoint(node_A, node_D) != Endpoint.CIRCLE:
                    continue

                if graph.get_endpoint(node_C, node_D) != Endpoint.CIRCLE:
                    continue

                if not is_arrow_point_allowed(node_D, node_B, graph, bk):
                    continue

                edge1 = graph.get_edge(node_D, node_B)
                graph.remove_edge(edge1)
                graph.add_edge(Edge(node_D, node_B, edge1.get_proximal_endpoint(node_D), Endpoint.ARROW))

                if verbose:
                    print("Orienting edge (Double triangle): " + graph.get_edge(node_D, node_B).__str__())

                changeFlag = True
    return changeFlag


def getPath(node_c: Node, previous) -> List[Node]:
    l = []
    node_p = previous[node_c]
    if node_p is not None:
        l.append(node_p)
    while node_p is not None:
        node_p = previous.get(node_p)
        if node_p is not None:
            l.append(node_p)
    return l


def doDdpOrientation(node_d: Node, node_a: Node, node_b: Node, node_c: Node, previous, graph: Graph, data,
                     independence_test_method, alpha: float, sep_sets: Dict[Tuple[int, int], Set[int]],
                     change_flag: bool, bk, verbose: bool = False) -> (bool, bool):
    if graph.is_adjacent_to(node_d, node_c):
        raise Exception("illegal argument!")
    path = getPath(node_d, previous)

    X, Y = graph.get_node_map()[node_d], graph.get_node_map()[node_c]
    condSet = tuple([graph.get_node_map()[nn] for nn in path])
    p_value = independence_test_method(X, Y, condSet)
    ind = p_value > alpha

    path2 = list(path)
    path2.remove(node_b)

    X, Y = graph.get_node_map()[node_d], graph.get_node_map()[node_c]
    condSet = tuple([graph.get_node_map()[nn2] for nn2 in path2])
    p_value2 = independence_test_method(X, Y, condSet)
    ind2 = p_value2 > alpha

    if not ind and not ind2:
        sep_set = sep_sets.get((graph.get_node_map()[node_d], graph.get_node_map()[node_c]))
        if verbose:
            message = "Sepset for d = " + node_d.get_name() + " and c = " + node_c.get_name() + " = [ "
            if sep_set is not None:
                for ss in sep_set:
                    message += graph.get_nodes()[ss].get_name() + " "
            message += "]"
            print(message)

        if sep_set is None:
            if verbose:
                print(
                    "Must be a sepset: " + node_d.get_name() + " and " + node_c.get_name() + "; they're non-adjacent.")
            return False, change_flag

        ind = sep_set.__contains__(graph.get_node_map()[node_b])

    if ind:
        edge = graph.get_edge(node_c, node_b)
        graph.remove_edge(edge)
        graph.add_edge(Edge(node_c, node_b, edge.get_proximal_endpoint(node_c), Endpoint.TAIL))

        if verbose:
            print(
                "Orienting edge (Definite discriminating path d = " + node_d.get_name() + "): " + graph.get_edge(node_b,
                                                                                                                 node_c).__str__())

        change_flag = True
        return True, change_flag
    else:
        if not is_arrow_point_allowed(node_a, node_b, graph, bk):
            return False, change_flag

        if not is_arrow_point_allowed(node_c, node_b, graph, bk):
            return False, change_flag

        edge1 = graph.get_edge(node_a, node_b)
        graph.remove_edge(edge1)
        graph.add_edge(Edge(node_a, node_b, edge1.get_proximal_endpoint(node_a), Endpoint.ARROW))

        edge2 = graph.get_edge(node_c, node_b)
        graph.remove_edge(edge2)
        graph.add_edge(Edge(node_c, node_b, edge2.get_proximal_endpoint(node_c), Endpoint.ARROW))

        if verbose:
            print(
                "Orienting collider (Definite discriminating path.. d = " + node_d.get_name() + "): " + node_a.get_name() + " *-> " + node_b.get_name() + " <-* " + node_c.get_name())

        change_flag = True
        return True, change_flag


def ddpOrient(node_a: Node, node_b: Node, node_c: Node, graph: Graph, maxPathLength: int, data: ndarray,
              independence_test_method, alpha: float, sep_sets: Dict[Tuple[int, int], Set[int]], change_flag: bool,
              bk: BackgroundKnowledge | None, verbose: bool = False) -> bool:
    Q = Queue()
    V = set()
    e = None
    distance = 0
    previous = {}

    cParents = graph.get_parents(node_c)

    Q.put(node_a)
    V.add(node_a)
    V.add(node_b)
    previous[node_a] = node_b

    while not Q.empty():
        node_t = Q.get_nowait()

        if e is None or e == node_t:
            e = node_t
            distance += 1
            if distance > 0 and distance > (1000 if maxPathLength == -1 else maxPathLength):
                return change_flag

        nodesInTo = graph.get_nodes_into(node_t, Endpoint.ARROW)

        for node_d in nodesInTo:
            if V.__contains__(node_d):
                continue

            previous[node_d] = node_t
            node_p = previous[node_t]

            if not graph.is_def_collider(node_d, node_t, node_p):
                continue

            previous[node_d] = node_t

            if not graph.is_adjacent_to(node_d, node_c) and node_d != node_c:
                res, change_flag = \
                    doDdpOrientation(node_d, node_a, node_b, node_c, previous, graph, data,
                                     independence_test_method, alpha, sep_sets, change_flag, bk, verbose)

                if res:
                    return change_flag

            if cParents.__contains__(node_d):
                Q.put(node_d)
                V.add(node_d)
    return change_flag


def ruleR4B(graph: Graph, maxPathLength: int, data: ndarray, independence_test_method, alpha: float,
            sep_sets: Dict[Tuple[int, int], Set[int]],
            change_flag: bool, bk: BackgroundKnowledge | None,
            verbose: bool = False) -> bool:
    nodes = graph.get_nodes()

    for node_b in nodes:
        possA = graph.get_nodes_out_of(node_b, Endpoint.ARROW)
        possC = graph.get_nodes_into(node_b, Endpoint.CIRCLE)

        for node_a in possA:
            for node_c in possC:
                if not graph.is_parent_of(node_a, node_c):
                    continue

                if graph.get_endpoint(node_b, node_c) != Endpoint.ARROW:
                    continue

                change_flag = ddpOrient(node_a, node_b, node_c, graph, maxPathLength, data, independence_test_method,
                                        alpha, sep_sets, change_flag, bk, verbose)
    return change_flag


def traverseSemiDirected(node: Node, edge: Edge) -> Node | None:
    if node == edge.get_node1():
        if edge.get_endpoint1() == Endpoint.TAIL or edge.get_endpoint1() == Endpoint.CIRCLE:
            return edge.get_node2()
    elif node == edge.get_node2():
        if edge.get_endpoint2() == Endpoint.TAIL or edge.get_endpoint2() == Endpoint.CIRCLE:
            return edge.get_node1()
    return None


def existsSemiDirectedPath(node_from: Node, node_to: Node, bound: int, graph: Graph) -> bool:
    Q = Queue()
    V = set()
    Q.put(node_from)
    V.add(node_from)
    node_e = None
    distance = 0

    while not Q.empty():
        node_t = Q.get_nowait()
        if node_t == node_to:
            return True

        if node_e == node_t:
            node_e = None
            distance += 1
            if distance > (1000 if bound == -1 else bound):
                return False

        for node_u in graph.get_adjacent_nodes(node_t):
            edge = graph.get_edge(node_t, node_u)
            node_c = traverseSemiDirected(node_t, edge)

            if node_c is None:
                continue

            if node_c == node_to:
                return True

            if not V.__contains__(node_c):
                V.add(node_c)
                Q.put(node_c)

                if node_e is None:
                    node_e = node_u

    return False


def visibleEdgeHelperVisit(graph: Graph, node_c: Node, node_a: Node, node_b: Node, path: List[Node]) -> bool:
    if path.__contains__(node_a):
        return False

    path.append(node_a)

    if node_a == node_b:
        return True

    for node_D in graph.get_nodes_into(node_a, Endpoint.ARROW):
        if graph.is_parent_of(node_D, node_c):
            return True

        if not graph.is_def_collider(node_D, node_c, node_a):
            continue
        elif not graph.is_parent_of(node_c, node_b):
            continue

        if visibleEdgeHelperVisit(graph, node_D, node_c, node_b, path):
            return True

    path.pop()
    return False


def visibleEdgeHelper(node_A: Node, node_B: Node, graph: Graph) -> bool:
    path = [node_A]

    for node_C in graph.get_nodes_into(node_A, Endpoint.ARROW):
        if graph.is_parent_of(node_C, node_A):
            return True

        if visibleEdgeHelperVisit(graph, node_C, node_A, node_B, path):
            return True

    return False


def defVisible(edge: Edge, graph: Graph) -> bool:
    if graph.contains_edge(edge):
        if edge.get_endpoint1() == Endpoint.TAIL:
            node_A = edge.get_node1()
            node_B = edge.get_node2()
        else:
            node_A = edge.get_node2()
            node_B = edge.get_node1()

        for node_C in graph.get_adjacent_nodes(node_A):
            if node_C != node_B and not graph.is_adjacent_to(node_C, node_B):
                e = graph.get_edge(node_C, node_A)

                if e.get_proximal_endpoint(node_A) == Endpoint.ARROW:
                    return True

        return visibleEdgeHelper(node_A, node_B, graph)
    else:
        raise Exception("Given edge is not in the graph.")


def get_color_edges(graph: Graph) -> List[Edge]:
    edges = graph.get_graph_edges()
    for edge in edges:
        if (edge.get_endpoint1() == Endpoint.TAIL and edge.get_endpoint2() == Endpoint.ARROW) or \
                (edge.get_endpoint1() == Endpoint.ARROW and edge.get_endpoint2() == Endpoint.TAIL):
            if edge.get_endpoint1() == Endpoint.TAIL:
                node_x = edge.get_node1()
                node_y = edge.get_node2()
            else:
                node_x = edge.get_node2()
                node_y = edge.get_node1()

            graph.remove_edge(edge)

            if not existsSemiDirectedPath(node_x, node_y, -1, graph):
                edge.properties.append(Edge.Property.dd)  # green
            else:
                edge.properties.append(Edge.Property.pd)

            graph.add_edge(edge)

            if defVisible(edge, graph):
                edge.properties.append(Edge.Property.nl)  # bold
            else:
                edge.properties.append(Edge.Property.pl)
    return edges


def fci(dataset: ndarray, independence_test_method: str=fisherz, alpha: float = 0.05, depth: int = -1,
        max_path_length: int = -1, verbose: bool = False, background_knowledge: BackgroundKnowledge | None = None) -> Tuple[Graph, List[Edge]]:
    """
    Perform Fast Causal Inference (FCI) algorithm for causal discovery

    Parameters
    ----------
    dataset: data set (numpy ndarray), shape (n_samples, n_features). The input data, where n_samples is the number of
            samples and n_features is the number of features.
    independence_test_method: str, name of the function of the independence test being used
            [fisherz, chisq, gsq, kci]
           - fisherz: Fisher's Z conditional independence test
           - chisq: Chi-squared conditional independence test
           - gsq: G-squared conditional independence test
           - kci: Kernel-based conditional independence test
    alpha: float, desired significance level of independence tests (p_value) in (0,1)
    depth: The depth for the fast adjacency search, or -1 if unlimited
    max_path_length: the maximum length of any discriminating path, or -1 if unlimited.
    verbose: True is verbose output should be printed or logged
    background_knowledge: background knowledge

    Returns
    -------
    graph : a CausalGraph object, where graph.graph[j,i]=1 and graph.graph[i,j]=-1 indicates  i --> j ,
                    graph.graph[i,j] = graph.graph[j,i] = -1 indicates i --- j,
                    graph.graph[i,j] = graph.graph[j,i] = 1 indicates i <-> j,
                    graph.graph[j,i]=1 and graph.graph[i,j]=2 indicates  i o-> j.
    edges : list
        Contains graph's edges properties.
        If edge.properties have the Property 'dd', then there is no latent confounder. Otherwise,
            there might be latent confounders.
        If edge.properties have the Property nl, then it is definitely direct. Otherwise,
            it is possibly direct.
    """

    if dataset.shape[0] < dataset.shape[1]:
        warnings.warn("The number of features is much larger than the sample size!")

    independence_test_method = CIT(dataset, method=independence_test_method)

    ## ------- check parameters ------------
    if (depth is None) or type(depth) != int:
        raise TypeError("'depth' must be 'int' type!")
    if (background_knowledge is not None) and type(background_knowledge) != BackgroundKnowledge:
        raise TypeError("'background_knowledge' must be 'BackgroundKnowledge' type!")
    if type(max_path_length) != int:
        raise TypeError("'max_path_length' must be 'int' type!")
    ## ------- end check parameters ------------


    nodes = []
    for i in range(dataset.shape[1]):
        node = GraphNode(f"X{i + 1}")
        node.add_attribute("id", i)
        nodes.append(node)

    # FAS (“Fast Adjacency Search”) is the adjacency search of the PC algorithm, used as a first step for the FCI algorithm.
    graph, sep_sets = fas(dataset, nodes, independence_test_method=independence_test_method, alpha=alpha,
                          knowledge=background_knowledge, depth=depth, verbose=verbose, show_progress=False)

    # reorient all edges with CIRCLE Endpoint
    ori_edges = graph.get_graph_edges()
    for ori_edge in ori_edges:
        graph.remove_edge(ori_edge)
        ori_edge.set_endpoint1(Endpoint.CIRCLE)
        ori_edge.set_endpoint2(Endpoint.CIRCLE)
        graph.add_edge(ori_edge)

    sp = SepsetsPossibleDsep(dataset, graph, independence_test_method, alpha, background_knowledge, depth,
                             max_path_length, verbose)

    rule0(graph, nodes, sep_sets, background_knowledge, verbose)

    waiting_to_deleted_edges = []

    for edge in graph.get_graph_edges():
        node_x = edge.get_node1()
        node_y = edge.get_node2()

        sep_set = sp.get_sep_set(node_x, node_y)

        if sep_set is not None:
            waiting_to_deleted_edges.append((node_x, node_y, sep_set))

    for waiting_to_deleted_edge in waiting_to_deleted_edges:
        dedge_node_x, dedge_node_y, dedge_sep_set = waiting_to_deleted_edge
        graph.remove_edge(graph.get_edge(dedge_node_x, dedge_node_y))
        sep_sets[(graph.node_map[dedge_node_x], graph.node_map[dedge_node_y])] = dedge_sep_set

        if verbose:
            message = "Possible DSEP Removed " + dedge_node_x.get_name() + " --- " + dedge_node_y.get_name() + " sepset = ["
            for ss in dedge_sep_set:
                message += graph.nodes[ss].get_name() + " "
            message += "]"
            print(message)

    reorientAllWith(graph, Endpoint.CIRCLE)
    rule0(graph, nodes, sep_sets, background_knowledge, verbose)

    change_flag = True
    first_time = True

    while change_flag:
        change_flag = False
        change_flag = rulesR1R2cycle(graph, background_knowledge, change_flag, verbose)
        change_flag = ruleR3(graph, sep_sets, background_knowledge, change_flag, verbose)

        if change_flag or (first_time and background_knowledge is not None and
                           len(background_knowledge.forbidden_rules_specs) > 0 and
                           len(background_knowledge.required_rules_specs) > 0 and
                           len(background_knowledge.tier_map.keys()) > 0):
            change_flag = ruleR4B(graph, max_path_length, dataset, independence_test_method, alpha, sep_sets,
                                  change_flag,
                                  background_knowledge, verbose)

            first_time = False

            if verbose:
                print("Epoch")

    graph.set_pag(True)

    edges = get_color_edges(graph)

    return graph, edges

def post_rule0_orient(skeleton: Graph, vstrucs: List[Tuple[int, int, int]], dataset: ndarray, independence_test_method: str=fisherz, alpha: float = 0.05, depth: int = -1,
        max_path_length: int = -1, verbose: bool = False, background_knowledge: BackgroundKnowledge | None = None) -> Tuple[Graph, List[Edge]]:
    """
    Perform Fast Causal Inference (FCI) algorithm for causal discovery

    Parameters
    ----------
    skeleton: adjacency matrix with oriented v-structures
    vstrucs: v-structures identified by ML4MAG
    dataset: data set (numpy ndarray), shape (n_samples, n_features). The input data, where n_samples is the number of
            samples and n_features is the number of features.
    independence_test_method: str, name of the function of the independence test being used
            [fisherz, chisq, gsq, kci]
           - fisherz: Fisher's Z conditional independence test
           - chisq: Chi-squared conditional independence test
           - gsq: G-squared conditional independence test
           - kci: Kernel-based conditional independence test
    alpha: float, desired significance level of independence tests (p_value) in (0,1)
    depth: The depth for the fast adjacency search, or -1 if unlimited
    max_path_length: the maximum length of any discriminating path, or -1 if unlimited.
    verbose: True is verbose output should be printed or logged
    background_knowledge: background knowledge

    Returns
    -------
    graph : a CausalGraph object, where graph.graph[j,i]=1 and graph.graph[i,j]=-1 indicates  i --> j ,
                    graph.graph[i,j] = graph.graph[j,i] = -1 indicates i --- j,
                    graph.graph[i,j] = graph.graph[j,i] = 1 indicates i <-> j,
                    graph.graph[j,i]=1 and graph.graph[i,j]=2 indicates  i o-> j.
    edges : list
        Contains graph's edges properties.
        If edge.properties have the Property 'dd', then there is no latent confounder. Otherwise,
            there might be latent confounders.
        If edge.properties have the Property nl, then it is definitely direct. Otherwise,
            it is possibly direct.
    """

    if dataset.shape[0] < dataset.shape[1]:
        warnings.warn("The number of features is much larger than the sample size!")

    independence_test_method = CIT(dataset, method=independence_test_method)

    ## ------- check parameters ------------
    if (depth is None) or type(depth) != int:
        raise TypeError("'depth' must be 'int' type!")
    if (background_knowledge is not None) and type(background_knowledge) != BackgroundKnowledge:
        raise TypeError("'background_knowledge' must be 'BackgroundKnowledge' type!")
    if type(max_path_length) != int:
        raise TypeError("'max_path_length' must be 'int' type!")
    ## ------- end check parameters ------------


    # nodes = []
    # for i in range(dataset.shape[1]):
    #     node = GraphNode(f"X{i + 1}")
    #     node.add_attribute("id", i)
    #     nodes.append(node)
    nodes = skeleton.get_nodes()
    # FAS (“Fast Adjacency Search”) is the adjacency search of the PC algorithm, used as a first step for the FCI algorithm.
    _, sep_sets = fas(dataset, nodes, independence_test_method=independence_test_method, alpha=alpha,
                          knowledge=background_knowledge, depth=depth, verbose=verbose)
            
    # reorient all edges with CIRCLE Endpoint
    ori_edges = skeleton.get_graph_edges()
    graph = GeneralGraph(nodes)
    for ori_edge in ori_edges:
        ori_edge.set_endpoint1(Endpoint.CIRCLE)
        ori_edge.set_endpoint2(Endpoint.CIRCLE)
        graph.add_edge(ori_edge)
    for vstruc in vstrucs:
        j, i, k = vstruc
        node_a = graph.get_nodes()[i]
        node_b = graph.get_nodes()[j]
        node_c = graph.get_nodes()[k]

        edge1 = graph.get_edge(node_a, node_b)
        graph.remove_edge(edge1)
        graph.add_edge(Edge(node_a, node_b, edge1.get_proximal_endpoint(node_a), Endpoint.ARROW))

        edge2 = graph.get_edge(node_c, node_b)
        graph.remove_edge(edge2)
        graph.add_edge(Edge(node_c, node_b, edge2.get_proximal_endpoint(node_c), Endpoint.ARROW))

    change_flag = True
    first_time = True

    while change_flag:
        change_flag = False
        change_flag = rulesR1R2cycle(graph, background_knowledge, change_flag, verbose)
        change_flag = ruleR3(graph, sep_sets, background_knowledge, change_flag, verbose)

        if change_flag or (first_time and background_knowledge is not None and
                           len(background_knowledge.forbidden_rules_specs) > 0 and
                           len(background_knowledge.required_rules_specs) > 0 and
                           len(background_knowledge.tier_map.keys()) > 0):
            change_flag = ruleR4B(graph, max_path_length, dataset, independence_test_method, alpha, sep_sets,
                                  change_flag,
                                  background_knowledge, verbose)

            first_time = False

            if verbose:
                print("Epoch")

    graph.set_pag(True)

    edges = get_color_edges(graph)

    return graph, edges
