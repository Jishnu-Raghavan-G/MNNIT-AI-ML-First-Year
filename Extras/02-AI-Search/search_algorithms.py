from collections import deque
import heapq


GRAPH = {
    "A": ["B", "C"],
    "B": ["D", "E"],
    "C": ["F", "G"],
    "D": [],
    "E": [],
    "F": [],
    "G": []
}


WEIGHTED_GRAPH = {
    "A": [("B", 2), ("C", 5)],
    "B": [("D", 4), ("E", 1)],
    "C": [("F", 2)],
    "D": [("G", 3)],
    "E": [("G", 5)],
    "F": [("G", 1)],
    "G": []
}


HEURISTIC = {
    "A": 7,
    "B": 5,
    "C": 4,
    "D": 3,
    "E": 2,
    "F": 1,
    "G": 0
}


def reconstruct_path(parent, goal):
    """Reconstruct a path from the parent dictionary."""
    path = []
    current = goal

    while current is not None:
        path.append(current)
        current = parent.get(current)

    path.reverse()
    return path


def bfs(graph, start, goal):
    """Breadth First Search."""
    queue = deque([start])
    visited = {start}
    parent = {start: None}

    while queue:
        current = queue.popleft()

        if current == goal:
            return reconstruct_path(parent, goal)

        for neighbor in graph.get(current, []):
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = current
                queue.append(neighbor)

    return None


def dfs(graph, start, goal):
    """Depth First Search."""
    stack = [start]
    visited = set()
    parent = {start: None}

    while stack:
        current = stack.pop()

        if current in visited:
            continue

        visited.add(current)

        if current == goal:
            return reconstruct_path(parent, goal)

        # Reverse insertion keeps the example's natural left-to-right order.
        for neighbor in reversed(graph.get(current, [])):
            if neighbor not in visited:
                if neighbor not in parent:
                    parent[neighbor] = current

                stack.append(neighbor)

    return None


def best_first_search(graph, start, goal, heuristic):
    """Greedy Best First Search using f(n) = h(n)."""
    priority_queue = []
    counter = 0

    heapq.heappush(
        priority_queue,
        (heuristic[start], counter, start)
    )

    visited = set()
    parent = {start: None}

    while priority_queue:
        _, _, current = heapq.heappop(priority_queue)

        if current in visited:
            continue

        visited.add(current)

        if current == goal:
            return reconstruct_path(parent, goal)

        for neighbor in graph.get(current, []):
            if neighbor not in visited:
                if neighbor not in parent:
                    parent[neighbor] = current

                counter += 1

                heapq.heappush(
                    priority_queue,
                    (heuristic[neighbor], counter, neighbor)
                )

    return None


def a_star(graph, start, goal, heuristic):
    """
    A* Search using:

        f(n) = g(n) + h(n)
    """
    priority_queue = []
    counter = 0

    g_cost = {start: 0}
    parent = {start: None}

    heapq.heappush(
        priority_queue,
        (heuristic[start], counter, start)
    )

    while priority_queue:
        current_f, _, current = heapq.heappop(priority_queue)

        expected_f = g_cost[current] + heuristic[current]

        # Ignore stale queue entries.
        if current_f != expected_f:
            continue

        if current == goal:
            return reconstruct_path(parent, goal), g_cost[goal]

        for neighbor, edge_cost in graph.get(current, []):
            tentative_g = g_cost[current] + edge_cost

            if (
                neighbor not in g_cost
                or tentative_g < g_cost[neighbor]
            ):
                g_cost[neighbor] = tentative_g
                parent[neighbor] = current

                counter += 1

                f_cost = (
                    tentative_g
                    + heuristic[neighbor]
                )

                heapq.heappush(
                    priority_queue,
                    (f_cost, counter, neighbor)
                )

    return None, None


def print_result(name, path):
    print(f"{name}:")
    if path is None:
        print("  No path found")
    else:
        print("  " + " -> ".join(path))


def main():
    start = "A"
    goal = "G"

    print("=== AI SEARCH ALGORITHMS ===\n")

    print_result(
        "Breadth First Search",
        bfs(GRAPH, start, goal)
    )

    print_result(
        "Depth First Search",
        dfs(GRAPH, start, goal)
    )

    print_result(
        "Best First Search",
        best_first_search(
            GRAPH,
            start,
            goal,
            HEURISTIC
        )
    )

    astar_path, astar_cost = a_star(
        WEIGHTED_GRAPH,
        start,
        goal,
        HEURISTIC
    )

    print("\nA* Search:")
    if astar_path is None:
        print("  No path found")
    else:
        print("  Path:", " -> ".join(astar_path))
        print("  Total cost:", astar_cost)

    print("\n=== SEARCH FUNCTIONS ===")
    print("BFS       -> queue -> shallowest first")
    print("DFS       -> stack -> deepest first")
    print("Best First -> h(n) -> estimated closeness")
    print("A*        -> g(n) + h(n) -> total estimated cost")


if __name__ == "__main__":
    main()   
