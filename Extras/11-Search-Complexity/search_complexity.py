from collections import deque
import heapq
import math
import time


# ============================================================
# LINEAR SEARCH
# ============================================================

def linear_search(array, target):
    for i, value in enumerate(array):
        if value == target:
            return i
    return -1


# ============================================================
# BINARY SEARCH
# ============================================================

def binary_search(array, target):
    left = 0
    right = len(array) - 1

    while left <= right:
        middle = (left + right) // 2

        if array[middle] == target:
            return middle

        if array[middle] < target:
            left = middle + 1
        else:
            right = middle - 1

    return -1


# ============================================================
# BFS
# ============================================================

def bfs(graph, start, goal=None):
    queue = deque([start])
    visited = {start}
    parent = {start: None}

    while queue:
        node = queue.popleft()

        if node == goal:
            break

        for neighbour in graph.get(node, []):
            if neighbour not in visited:
                visited.add(neighbour)
                parent[neighbour] = node
                queue.append(neighbour)

    if goal is None:
        return list(visited)

    if goal not in parent:
        return None

    path = []
    current = goal

    while current is not None:
        path.append(current)
        current = parent[current]

    path.reverse()
    return path


# ============================================================
# DFS
# ============================================================

def dfs(graph, start, goal=None):
    stack = [start]
    visited = set()
    parent = {start: None}

    while stack:
        node = stack.pop()

        if node in visited:
            continue

        visited.add(node)

        if node == goal:
            break

        for neighbour in reversed(graph.get(node, [])):
            if neighbour not in visited:
                if neighbour not in parent:
                    parent[neighbour] = node
                stack.append(neighbour)

    if goal is None:
        return list(visited)

    if goal not in visited:
        return None

    path = []
    current = goal

    while current is not None:
        path.append(current)
        current = parent[current]

    path.reverse()
    return path


# ============================================================
# UNIFORM-COST SEARCH
# ============================================================

def uniform_cost_search(graph, start, goal):
    priority_queue = [(0, start)]
    best_cost = {start: 0}
    parent = {start: None}

    while priority_queue:
        cost, node = heapq.heappop(priority_queue)

        if cost != best_cost[node]:
            continue

        if node == goal:
            path = []
            current = goal

            while current is not None:
                path.append(current)
                current = parent[current]

            path.reverse()
            return path, cost

        for neighbour, edge_cost in graph.get(node, []):
            new_cost = cost + edge_cost

            if (
                neighbour not in best_cost
                or new_cost < best_cost[neighbour]
            ):
                best_cost[neighbour] = new_cost
                parent[neighbour] = node

                heapq.heappush(
                    priority_queue,
                    (new_cost, neighbour)
                )

    return None, math.inf


# ============================================================
# GREEDY BEST-FIRST SEARCH
# ============================================================

def greedy_best_first(graph, start, goal, heuristic):
    priority_queue = [(heuristic[start], start)]
    visited = set()
    parent = {start: None}

    while priority_queue:
        _, node = heapq.heappop(priority_queue)

        if node in visited:
            continue

        visited.add(node)

        if node == goal:
            path = []
            current = goal

            while current is not None:
                path.append(current)
                current = parent[current]

            path.reverse()
            return path

        for neighbour in graph.get(node, []):
            if neighbour not in visited:
                if neighbour not in parent:
                    parent[neighbour] = node

                heapq.heappush(
                    priority_queue,
                    (heuristic[neighbour], neighbour)
                )

    return None


# ============================================================
# A* SEARCH
# ============================================================

def a_star(graph, start, goal, heuristic):
    priority_queue = [(heuristic[start], 0, start)]
    best_cost = {start: 0}
    parent = {start: None}

    while priority_queue:
        _, cost, node = heapq.heappop(priority_queue)

        if cost != best_cost[node]:
            continue

        if node == goal:
            path = []
            current = goal

            while current is not None:
                path.append(current)
                current = parent[current]

            path.reverse()
            return path, cost

        for neighbour, edge_cost in graph.get(node, []):
            new_cost = cost + edge_cost

            if (
                neighbour not in best_cost
                or new_cost < best_cost[neighbour]
            ):
                best_cost[neighbour] = new_cost
                parent[neighbour] = node

                f = new_cost + heuristic[neighbour]

                heapq.heappush(
                    priority_queue,
                    (f, new_cost, neighbour)
                )

    return None, math.inf


# ============================================================
# TREE NODE
# ============================================================

class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None


# ============================================================
# TREE TRAVERSALS
# ============================================================

def preorder(root):
    if root is None:
        return []

    return (
        [root.value]
        + preorder(root.left)
        + preorder(root.right)
    )


def inorder(root):
    if root is None:
        return []

    return (
        inorder(root.left)
        + [root.value]
        + inorder(root.right)
    )


def postorder(root):
    if root is None:
        return []

    return (
        postorder(root.left)
        + postorder(root.right)
        + [root.value]
    )


def tree_bfs(root):
    if root is None:
        return []

    queue = deque([root])
    result = []

    while queue:
        node = queue.popleft()
        result.append(node.value)

        if node.left:
            queue.append(node.left)

        if node.right:
            queue.append(node.right)

    return result


# ============================================================
# 8-PUZZLE MANHATTAN DISTANCE
# ============================================================

def manhattan_distance(state, goal):
    goal_position = {
        value: index
        for index, value in enumerate(goal)
    }

    distance = 0

    for index, value in enumerate(state):
        if value == 0:
            continue

        current_row, current_col = divmod(index, 3)

        goal_index = goal_position[value]
        goal_row, goal_col = divmod(goal_index, 3)

        distance += abs(current_row - goal_row)
        distance += abs(current_col - goal_col)

    return distance


# ============================================================
# COMPLEXITY TABLE
# ============================================================

def print_complexity_table():
    print("\nComplexity Summary")
    print("-" * 72)

    rows = [
        ("Linear Search", "O(n)", "O(1)"),
        ("Binary Search", "O(log n)", "O(1) iterative"),
        ("BFS Tree Search", "O(b^d)", "O(b^d)"),
        ("DFS Tree Search", "O(b^m)", "O(bm)"),
        ("Tree Traversal", "O(n)", "O(h) DFS / O(w) BFS"),
        ("Graph BFS", "O(V + E)", "O(V)"),
        ("Graph DFS", "O(V + E)", "O(V)"),
    ]

    print(f"{'Algorithm':25} {'Time':20} {'Space'}")
    print("-" * 72)

    for name, time_complexity, space_complexity in rows:
        print(
            f"{name:25} "
            f"{time_complexity:20} "
            f"{space_complexity}"
        )


# ============================================================
# MAIN DEMONSTRATION
# ============================================================

if __name__ == "__main__":

    print("=" * 72)
    print("SEARCH COMPLEXITY DEMONSTRATION")
    print("=" * 72)

    # --------------------------------------------------------
    # 1. Linear Search
    # --------------------------------------------------------

    array = [10, 20, 30, 40, 50]

    print("\n1. Linear Search")
    print("Array:", array)
    print("Target: 40")
    print("Index:", linear_search(array, 40))

    # --------------------------------------------------------
    # 2. Binary Search
    # --------------------------------------------------------

    sorted_array = [10, 20, 30, 40, 50, 60, 70]

    print("\n2. Binary Search")
    print("Sorted array:", sorted_array)
    print("Target: 60")
    print("Index:", binary_search(sorted_array, 60))

    # --------------------------------------------------------
    # 3. BFS
    # --------------------------------------------------------

    graph = {
        "A": ["B", "C"],
        "B": ["D", "E"],
        "C": ["F"],
        "D": [],
        "E": [],
        "F": [],
    }

    print("\n3. BFS")
    print("BFS path A -> F:")
    print(bfs(graph, "A", "F"))

    # --------------------------------------------------------
    # 4. DFS
    # --------------------------------------------------------

    print("\n4. DFS")
    print("DFS path A -> F:")
    print(dfs(graph, "A", "F"))

    # --------------------------------------------------------
    # 5. Uniform-Cost Search
    # --------------------------------------------------------

    weighted_graph = {
        "A": [("B", 1), ("C", 4)],
        "B": [("D", 2), ("E", 5)],
        "C": [("D", 1)],
        "D": [("E", 1)],
        "E": [],
    }

    print("\n5. Uniform-Cost Search")

    path, cost = uniform_cost_search(
        weighted_graph,
        "A",
        "E"
    )

    print("Path:", path)
    print("Cost:", cost)

    # --------------------------------------------------------
    # 6. Greedy Best-First Search
    # --------------------------------------------------------

    heuristic = {
        "A": 5,
        "B": 4,
        "C": 2,
        "D": 1,
        "E": 0,
    }

    print("\n6. Greedy Best-First Search")

    greedy_path = greedy_best_first(
        weighted_graph,
        "A",
        "E",
        heuristic
    )

    print("Path:", greedy_path)

    # --------------------------------------------------------
    # 7. A* Search
    # --------------------------------------------------------

    print("\n7. A* Search")

    path, cost = a_star(
        weighted_graph,
        "A",
        "E",
        heuristic
    )

    print("Path:", path)
    print("Cost:", cost)

    # --------------------------------------------------------
    # 8. Tree Traversals
    # --------------------------------------------------------

    root = TreeNode(5)

    root.left = TreeNode(3)
    root.right = TreeNode(7)

    root.left.left = TreeNode(2)
    root.left.right = TreeNode(4)

    root.right.left = TreeNode(6)
    root.right.right = TreeNode(8)

    print("\n8. Tree Traversals")

    print("Preorder :", preorder(root))
    print("Inorder  :", inorder(root))
    print("Postorder:", postorder(root))
    print("BFS      :", tree_bfs(root))

    # --------------------------------------------------------
    # 9. 8-Puzzle Manhattan Distance
    # --------------------------------------------------------

    goal = (
        1, 2, 3,
        4, 5, 6,
        7, 8, 0
    )

    state = (
        1, 2, 3,
        4, 5, 6,
        0, 7, 8
    )

    print("\n9. 8-Puzzle Manhattan Distance")

    print("State:", state)
    print("Goal :", goal)
    print(
        "Manhattan distance:",
        manhattan_distance(state, goal)
    )

    # --------------------------------------------------------
    # 10. Complexity Table
    # --------------------------------------------------------

    print_complexity_table()

    # --------------------------------------------------------
    # 11. Simple Timing Comparison
    # --------------------------------------------------------

    print("\n11. Simple Search Timing")

    large_array = list(range(1_000_000))
    target = 999_999

    start = time.perf_counter()
    linear_search(large_array, target)
    linear_time = time.perf_counter() - start

    start = time.perf_counter()
    binary_search(large_array, target)
    binary_time = time.perf_counter() - start

    print(f"Linear search time: {linear_time:.8f} seconds")
    print(f"Binary search time: {binary_time:.8f} seconds")

    print("\nComplexity Reminder:")
    print("Linear Search -> O(n)")
    print("Binary Search -> O(log n)")
    print("BFS           -> O(b^d)")
    print("DFS           -> O(b^m)")
    print("Tree Traversal-> O(n)")
    print("A*            -> f(n) = g(n) + h(n)")
