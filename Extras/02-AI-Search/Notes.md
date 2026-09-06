# AI Search Algorithms

## 1. Introduction

Artificial Intelligence often requires an agent to find a sequence of actions that transforms an initial state into a desired goal state.

Examples:

- Finding a route from one city to another.
- Solving a puzzle.
- Finding a path through a maze.
- Planning actions to reach a goal.
- Searching a game state space.

Search algorithms systematically explore possible states until they find a goal.

---

## 2. State

A state represents a particular configuration of a problem.

Examples:

### Maze

A state can be the current cell.

### 8-Puzzle

A state is the current arrangement of the tiles.

### Route Finding

A state can be the current city.

---

## 3. Initial State

The initial state is where the search begins.

Example:

Start city = A

The search starts from A.

---

## 4. Goal State

The goal state is the desired final state.

Example:

Start = A

Goal = G

The search tries to find a path:

A → ... → G

---

## 5. State Space

The state space is the collection of possible states that can be reached from the initial state.

A search algorithm explores this state space.

A large state space can make search computationally expensive.

---

## 6. Search Tree and Search Graph

### Search Tree

A search tree represents states as nodes and possible actions as branches.

Example:

A
├── B
│   ├── D
│   └── E
└── C
    ├── F
    └── G

### Search Graph

A graph can contain connections between states that are not necessarily arranged as a tree.

Graphs can contain cycles.

Therefore, graph-search algorithms generally need a mechanism to remember visited states.

---

## 7. Node

A search node can contain information such as:

- Current state
- Parent node
- Action used to reach the state
- Path cost
- Depth

The parent relationship allows the final path to be reconstructed after reaching the goal.

---

## 8. Queue

A queue follows:

FIFO

First In, First Out.

Example:

A → B → C

If A is inserted first, A is removed first.

Queues are important for Breadth First Search.

---

## 9. Stack

A stack follows:

LIFO

Last In, First Out.

Example:

A → B → C

C is removed first.

Stacks are important for Depth First Search.

---

## 10. Priority Queue

A priority queue removes the element with the highest priority according to a priority value.

Priority queues are useful in heuristic search and A*.

For example:

Node A: priority 8
Node B: priority 3
Node C: priority 5

The node with the smallest priority value can be selected first when the algorithm defines lower values as better.

---

# 11. Uninformed Search

Uninformed search does not use additional information about how close a state is to the goal.

Important uninformed algorithms include:

- Breadth First Search
- Depth First Search

---

# 12. Breadth First Search

Breadth First Search explores nodes level by level.

It uses a queue.

Example:

A
├── B
├── C
└── D

BFS explores:

A → B → C → D → ...

Then it explores the next level.

---

## BFS Procedure

1. Put the initial state into the queue.
2. Remove the first state.
3. Check whether it is the goal.
4. Expand its children.
5. Add unvisited children to the queue.
6. Repeat until the goal is found or the queue becomes empty.

---

## BFS Pseudocode

    queue = [start]
    visited = {start}

    while queue is not empty:
        node = remove_front(queue)

        if node is goal:
            return solution

        for child in children(node):
            if child not in visited:
                add child to visited
                add child to queue

    return failure

---

## BFS Properties

BFS explores the shallowest available nodes first.

For an unweighted graph, BFS can find a shortest path in terms of number of edges when the graph is properly represented and visited states are handled.

BFS can require significant memory because it stores a frontier of nodes.

---

# 13. Depth First Search

Depth First Search explores one branch as deeply as possible before backtracking.

It uses a stack or recursion.

Example:

A
├── B
│   ├── D
│   └── E
└── C

DFS may explore:

A → B → D → E → C

depending on the ordering of children.

---

## DFS Procedure

1. Put the initial state into the stack.
2. Remove the top state.
3. Check whether it is the goal.
4. Expand its children.
5. Add unvisited children to the stack.
6. Continue until the goal is found or the stack becomes empty.

---

## DFS Pseudocode

    stack = [start]
    visited = set()

    while stack is not empty:
        node = remove_top(stack)

        if node in visited:
            continue

        add node to visited

        if node is goal:
            return solution

        for child in children(node):
            add child to stack

    return failure

---

## DFS Properties

DFS can use less memory than BFS in many search spaces because it primarily maintains the current path and frontier.

However, DFS can follow a very deep or unproductive branch before finding a solution.

DFS does not generally guarantee a shortest path.

---

# 14. BFS vs DFS

| Property | BFS | DFS |
|---|---|---|
| Main data structure | Queue | Stack |
| Search order | Level by level | Depth first |
| Shortest path in unweighted graph | Yes, under standard conditions | No |
| Memory usage | Usually high | Usually lower |
| Can get trapped in deep branch | No in finite standard BFS | Yes |
| Main idea | Explore broadly | Explore deeply |

---

# 15. Heuristic Search

Heuristic search uses additional information to decide which state should be explored next.

A heuristic function is commonly represented as:

h(n)

where h(n) estimates the cost from node n to the goal.

A good heuristic can reduce the number of states explored.

---

# 16. Heuristic Function

The heuristic function:

h(n)

estimates how far node n is from the goal.

Example:

In route finding:

h(n) = estimated distance from current city to destination

In the 8-puzzle:

h(n) can be the number of misplaced tiles or Manhattan distance.

The heuristic is an estimate, not necessarily the exact remaining cost.

---

# 17. Best First Search

Best First Search selects the node that appears most promising according to a priority function.

A common greedy form uses:

f(n) = h(n)

Therefore, the node with the smallest heuristic estimate is selected first.

---

## Best First Search Idea

If:

A: h(A) = 8
B: h(B) = 3
C: h(C) = 5

the algorithm prefers:

B

because B has the smallest heuristic value.

---

## Important Limitation

Greedy Best First Search focuses on estimated distance to the goal.

It does not necessarily consider the cost already spent to reach the node.

Therefore, it does not generally guarantee an optimal solution.

---

# 18. A* Search

A* combines the cost already travelled with an estimate of the remaining cost.

The evaluation function is:

f(n) = g(n) + h(n)

Where:

g(n) = actual cost from the start to n

h(n) = estimated cost from n to the goal

f(n) = estimated total cost of a solution through n

---

## Example

Suppose:

g(A) = 4
h(A) = 6

Then:

f(A) = 4 + 6
f(A) = 10

A* compares nodes using their f values.

---

# 19. A* Procedure

1. Put the initial node in the priority queue.
2. Calculate its f value.
3. Select the node with the smallest f value.
4. Check whether it is the goal.
5. Expand the node.
6. Calculate g, h and f for successors.
7. Update nodes when a cheaper path is discovered.
8. Continue until the goal is reached.

---

## A* Pseudocode

    open_set = priority_queue()
    open_set.add(start)

    g[start] = 0
    f[start] = h(start)

    while open_set is not empty:

        current = node_with_smallest_f()

        if current is goal:
            return path

        for neighbor in neighbors(current):

            tentative_g = g[current] + cost(current, neighbor)

            if tentative_g < g[neighbor]:

                parent[neighbor] = current
                g[neighbor] = tentative_g
                f[neighbor] = g[neighbor] + h(neighbor)

                add neighbor to open_set

    return failure

---

# 20. A* Optimality

A* can find an optimal solution when the heuristic satisfies appropriate conditions.

A commonly discussed condition is admissibility.

---

## Admissible Heuristic

A heuristic is admissible if it never overestimates the true remaining cost.

Therefore:

h(n) ≤ h*(n)

where h*(n) is the actual minimum cost from n to the goal.

An admissible heuristic is optimistic.

---

# 21. Consistent Heuristic

A heuristic is consistent if, for every edge from n to n':

h(n) ≤ cost(n,n') + h(n')

Consistency is also called monotonicity.

A consistent heuristic is admissible under the usual nonnegative-cost assumptions.

---

# 22. 8-Puzzle

The 8-puzzle consists of:

- Eight numbered tiles.
- One blank space.
- A 3 × 3 board.

The objective is to transform an initial arrangement into a target arrangement.

Example goal:

1 2 3
4 5 6
7 8 _

A state can be represented by the complete board arrangement.

---

# 23. Actions in 8-Puzzle

The blank tile can move:

- Up
- Down
- Left
- Right

when the corresponding move is valid.

Each move produces a new state.

---

# 24. Heuristics for 8-Puzzle

Two common heuristic ideas are:

### Misplaced Tiles

Count the number of tiles that are not in their goal positions.

### Manhattan Distance

For each tile:

distance = |current_row - goal_row|
         + |current_column - goal_column|

The total Manhattan distance is the sum over relevant tiles.

---

# 25. A* for 8-Puzzle

A* can use:

f(n) = g(n) + h(n)

For example:

g(n) = number of moves made so far

h(n) = Manhattan distance of the current state

Then:

f(n) = moves_so_far + Manhattan_distance

The state with the smallest f value is expanded next.

---

# 26. Search Algorithm Comparison

| Algorithm | Uses Heuristic | Main Selection Rule |
|---|---|---|
| BFS | No | Smallest depth |
| DFS | No | Deepest available |
| Best First | Yes | Smallest h(n) |
| A* | Yes | Smallest g(n) + h(n) |

---

# 27. Common Mistakes

### Mistake 1: Confusing BFS and DFS

BFS uses a queue.

DFS uses a stack or recursion.

### Mistake 2: Assuming DFS gives the shortest path

DFS does not generally guarantee the shortest path.

### Mistake 3: Confusing Best First Search and A*

Greedy Best First Search commonly uses:

f(n) = h(n)

A* uses:

f(n) = g(n) + h(n)

### Mistake 4: Thinking Every Heuristic Is Admissible

A heuristic must satisfy the appropriate condition to be admissible.

### Mistake 5: Ignoring Repeated States

Graphs can contain cycles.

Visited-state handling prevents unnecessary repeated exploration.

---

# 28. Exam Summary

### BFS

Queue → level by level → shortest path in unweighted graphs under standard conditions.

### DFS

Stack/recursion → depth first → does not generally guarantee shortest path.

### Best First Search

Uses heuristic:

f(n) = h(n)

### A*

Uses:

f(n) = g(n) + h(n)

### Admissibility

Heuristic never overestimates true remaining cost.

### Consistency

h(n) ≤ cost(n,n') + h(n')

### 8-Puzzle

State = board configuration.

Actions = valid blank movements.

Heuristic = misplaced tiles or Manhattan distance.

---

# 29. Final Mental Model

Search problems can be understood as:

Initial State
    ↓
Generate possible actions
    ↓
Generate successor states
    ↓
Choose which state to explore
    ↓
Continue searching
    ↓
Goal State

The major difference between search algorithms is how they choose the next state.

BFS → shallowest

DFS → deepest

Best First → smallest heuristic

A* → smallest actual cost + estimated remaining cost
