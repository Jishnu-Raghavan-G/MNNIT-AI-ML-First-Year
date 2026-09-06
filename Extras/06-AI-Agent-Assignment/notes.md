# Artificial Agents Assignment

## 1. Project Overview

This assignment introduces Artificial Intelligence agents and different approaches to designing agents.

An agent perceives its environment and takes actions based on those perceptions.

A simple representation is:

    Environment
         ↓
      Percepts
         ↓
       Agent
         ↓
      Actions
         ↓
    Environment

The agent continuously interacts with its environment.

---

## 2. What Is an Artificial Agent?

An artificial agent is a system that:

- Perceives its environment.
- Processes available information.
- Selects an action.
- Acts on the environment.

Examples include:

- Robot navigation systems.
- Game-playing programs.
- Recommendation systems.
- Autonomous vehicles.
- Software assistants.

---

## 3. Agent and Environment

The environment contains everything outside the agent that can affect its behavior.

The agent receives information from the environment through percepts.

It then selects an action.

Example:

    Environment:
    Room with obstacles

    Percept:
    Obstacle detected ahead

    Action:
    Turn left

---

## 4. Rational Agent

A rational agent selects an action that is expected to achieve the best outcome according to its available information and performance measure.

The agent does not necessarily know everything about the environment.

It makes decisions using the information available to it.

---

## 5. Types of Artificial Agents

Important agent types include:

1. Simple Reflex Agent
2. Model-Based Reflex Agent
3. Goal-Based Agent
4. Utility-Based Agent

These represent increasing levels of decision-making complexity.

---

# 6. Simple Reflex Agent

A Simple Reflex Agent selects actions using the current percept.

It follows condition-action rules.

General structure:

    IF condition
    THEN action

Example:

    IF obstacle_detected
    THEN turn_left

The agent does not need to reason about the complete history of the environment.

---

## Example

Suppose a cleaning robot detects dirt.

Percept:

    dirt_detected = True

Rule:

    IF dirt_detected
    THEN clean

Action:

    Clean the location

---

## Advantages

- Simple.
- Fast.
- Easy to implement.
- Suitable for simple environments.

## Limitations

- Limited decision-making.
- Cannot handle situations requiring substantial memory.
- May perform poorly when the current percept is insufficient.

---

# 7. Model-Based Reflex Agent

A Model-Based Reflex Agent maintains an internal representation of the environment.

It uses:

- Current percept.
- Previous information.
- Internal state.
- A model of how the environment changes.

General idea:

    Percept
       ↓
    Internal State
       ↓
    Rule / Decision
       ↓
    Action

This allows the agent to work in partially observable environments better than a purely reflex agent.

---

## Example

A robot enters a room and remembers that it already cleaned one area.

Even if that area is not currently visible, the internal state can help the robot decide what to do next.

---

# 8. Goal-Based Agent

A Goal-Based Agent selects actions based on a desired goal.

The agent considers possible actions and determines which actions can help it reach the goal.

Example:

Goal:

    Reach destination B

Possible actions:

    Move left
    Move right
    Move forward
    Move backward

The agent searches for a sequence of actions that reaches B.

---

## Goal-Based Decision Process

    Current State
         ↓
    Possible Actions
         ↓
    Search / Planning
         ↓
    Goal State
         ↓
    Selected Action

Search algorithms such as BFS, DFS and A* can be used in goal-directed problems.

---

# 9. Utility-Based Agent

A Utility-Based Agent considers how desirable different outcomes are.

Instead of asking only:

    "Can I reach the goal?"

it can ask:

    "Which possible outcome is better?"

A utility function assigns a value to outcomes.

Higher utility generally represents a more desirable outcome.

---

## Example

A navigation agent may have several possible routes:

Route A:

    Travel time = 20 minutes
    Cost = low
    Utility = 70

Route B:

    Travel time = 15 minutes
    Cost = high
    Utility = 60

Route C:

    Travel time = 18 minutes
    Cost = medium
    Utility = 85

The agent can select Route C because it has the highest utility according to its utility function.

---

# 10. Comparison of Agent Types

| Agent Type | Main Idea | Memory / Internal State | Goal | Utility |
|---|---|---|---|---|
| Simple Reflex | Current percept → action | No substantial internal state | Not required | No |
| Model-Based Reflex | Percept + internal state → action | Yes | Not necessarily | No |
| Goal-Based | Choose actions to reach goal | Often required | Yes | No |
| Utility-Based | Choose best outcome | Yes | Usually involved | Yes |

---

# 11. Environment

An agent's behavior depends strongly on the environment.

Important environmental characteristics include:

- Fully observable vs partially observable
- Deterministic vs stochastic
- Static vs dynamic
- Discrete vs continuous
- Single-agent vs multi-agent

The design of an agent should match the environment in which it operates.

---

# 12. Agent Architecture

A basic agent architecture can be represented as:

    Sensors
       ↓
    Percepts
       ↓
    Agent Program
       ↓
    Decision
       ↓
    Actuators
       ↓
    Environment

Sensors provide information about the environment.

The agent program processes the percepts.

Actuators execute the selected action.

---

# 13. Example: Vacuum Cleaner Agent

Consider a two-room environment:

    [Room A] [Room B]

The agent can:

- Move left.
- Move right.
- Clean.

### Simple Reflex Rules

    IF current_room_is_dirty
        CLEAN

    ELSE IF current_room == A
        MOVE_RIGHT

    ELSE
        MOVE_LEFT

The rules directly map percepts to actions.

---

# 14. Example: Goal-Based Navigation Agent

Suppose:

    Start = A
    Goal = G

The environment is:

    A → B → D → G
     \
      → C → F → G

A goal-based agent can search through the possible states and find a path to G.

Search algorithms can be used to determine the sequence of actions.

---

# 15. Example: Utility-Based Navigation Agent

Suppose an agent has multiple possible routes.

It may consider:

- Distance.
- Time.
- Cost.
- Safety.

A utility function can combine these factors.

For example:

    utility =
        - distance
        - travel_time
        - cost
        + safety

The exact utility function depends on the application's objectives.

The agent selects the action sequence with the best expected utility.

---

# 16. Assignment Questions

### Question 1

Define an Artificial Agent.

### Question 2

Explain the difference between an agent and its environment.

### Question 3

Explain a Simple Reflex Agent with an example.

### Question 4

Explain a Model-Based Reflex Agent.

### Question 5

Explain a Goal-Based Agent.

### Question 6

Explain a Utility-Based Agent.

### Question 7

Compare the four major agent types.

### Question 8

Design an agent for a simple vacuum-cleaner environment.

### Question 9

Explain how search algorithms can be used by a Goal-Based Agent.

### Question 10

Explain why a Utility-Based Agent can be preferable when multiple valid solutions exist.

---

# 17. Learning Objectives

After completing this assignment, you should understand:

- What an artificial agent is.
- How agents interact with environments.
- What percepts and actions are.
- How Simple Reflex Agents work.
- How Model-Based Reflex Agents maintain internal state.
- How Goal-Based Agents use goals.
- How Utility-Based Agents compare outcomes.
- How search algorithms can support intelligent agents.

---

# 18. Key Takeaway

The progression can be remembered as:

Simple Reflex:

    "What do I perceive right now?"

Model-Based:

    "What do I perceive, and what do I remember?"

Goal-Based:

    "What action helps me reach my goal?"

Utility-Based:

    "Which available outcome is best?"# Artificial Agents Assignment

## 1. Project Overview

This assignment introduces Artificial Intelligence agents and different approaches to designing agents.

An agent perceives its environment and takes actions based on those perceptions.

A simple representation is:

    Environment
         ↓
      Percepts
         ↓
       Agent
         ↓
      Actions
         ↓
    Environment

The agent continuously interacts with its environment.

---

## 2. What Is an Artificial Agent?

An artificial agent is a system that:

- Perceives its environment.
- Processes available information.
- Selects an action.
- Acts on the environment.

Examples include:

- Robot navigation systems.
- Game-playing programs.
- Recommendation systems.
- Autonomous vehicles.
- Software assistants.

---

## 3. Agent and Environment

The environment contains everything outside the agent that can affect its behavior.

The agent receives information from the environment through percepts.

It then selects an action.

Example:

    Environment:
    Room with obstacles

    Percept:
    Obstacle detected ahead

    Action:
    Turn left

---

## 4. Rational Agent

A rational agent selects an action that is expected to achieve the best outcome according to its available information and performance measure.

The agent does not necessarily know everything about the environment.

It makes decisions using the information available to it.

---

## 5. Types of Artificial Agents

Important agent types include:

1. Simple Reflex Agent
2. Model-Based Reflex Agent
3. Goal-Based Agent
4. Utility-Based Agent

These represent increasing levels of decision-making complexity.

---

# 6. Simple Reflex Agent

A Simple Reflex Agent selects actions using the current percept.

It follows condition-action rules.

General structure:

    IF condition
    THEN action

Example:

    IF obstacle_detected
    THEN turn_left

The agent does not need to reason about the complete history of the environment.

---

## Example

Suppose a cleaning robot detects dirt.

Percept:

    dirt_detected = True

Rule:

    IF dirt_detected
    THEN clean

Action:

    Clean the location

---

## Advantages

- Simple.
- Fast.
- Easy to implement.
- Suitable for simple environments.

## Limitations

- Limited decision-making.
- Cannot handle situations requiring substantial memory.
- May perform poorly when the current percept is insufficient.

---

# 7. Model-Based Reflex Agent

A Model-Based Reflex Agent maintains an internal representation of the environment.

It uses:

- Current percept.
- Previous information.
- Internal state.
- A model of how the environment changes.

General idea:

    Percept
       ↓
    Internal State
       ↓
    Rule / Decision
       ↓
    Action

This allows the agent to work in partially observable environments better than a purely reflex agent.

---

## Example

A robot enters a room and remembers that it already cleaned one area.

Even if that area is not currently visible, the internal state can help the robot decide what to do next.

---

# 8. Goal-Based Agent

A Goal-Based Agent selects actions based on a desired goal.

The agent considers possible actions and determines which actions can help it reach the goal.

Example:

Goal:

    Reach destination B

Possible actions:

    Move left
    Move right
    Move forward
    Move backward

The agent searches for a sequence of actions that reaches B.

---

## Goal-Based Decision Process

    Current State
         ↓
    Possible Actions
         ↓
    Search / Planning
         ↓
    Goal State
         ↓
    Selected Action

Search algorithms such as BFS, DFS and A* can be used in goal-directed problems.

---

# 9. Utility-Based Agent

A Utility-Based Agent considers how desirable different outcomes are.

Instead of asking only:

    "Can I reach the goal?"

it can ask:

    "Which possible outcome is better?"

A utility function assigns a value to outcomes.

Higher utility generally represents a more desirable outcome.

---

## Example

A navigation agent may have several possible routes:

Route A:

    Travel time = 20 minutes
    Cost = low
    Utility = 70

Route B:

    Travel time = 15 minutes
    Cost = high
    Utility = 60

Route C:

    Travel time = 18 minutes
    Cost = medium
    Utility = 85

The agent can select Route C because it has the highest utility according to its utility function.

---

# 10. Comparison of Agent Types

| Agent Type | Main Idea | Memory / Internal State | Goal | Utility |
|---|---|---|---|---|
| Simple Reflex | Current percept → action | No substantial internal state | Not required | No |
| Model-Based Reflex | Percept + internal state → action | Yes | Not necessarily | No |
| Goal-Based | Choose actions to reach goal | Often required | Yes | No |
| Utility-Based | Choose best outcome | Yes | Usually involved | Yes |

---

# 11. Environment

An agent's behavior depends strongly on the environment.

Important environmental characteristics include:

- Fully observable vs partially observable
- Deterministic vs stochastic
- Static vs dynamic
- Discrete vs continuous
- Single-agent vs multi-agent

The design of an agent should match the environment in which it operates.

---

# 12. Agent Architecture

A basic agent architecture can be represented as:

    Sensors
       ↓
    Percepts
       ↓
    Agent Program
       ↓
    Decision
       ↓
    Actuators
       ↓
    Environment

Sensors provide information about the environment.

The agent program processes the percepts.

Actuators execute the selected action.

---

# 13. Example: Vacuum Cleaner Agent

Consider a two-room environment:

    [Room A] [Room B]

The agent can:

- Move left.
- Move right.
- Clean.

### Simple Reflex Rules

    IF current_room_is_dirty
        CLEAN

    ELSE IF current_room == A
        MOVE_RIGHT

    ELSE
        MOVE_LEFT

The rules directly map percepts to actions.

---

# 14. Example: Goal-Based Navigation Agent

Suppose:

    Start = A
    Goal = G

The environment is:

    A → B → D → G
     \
      → C → F → G

A goal-based agent can search through the possible states and find a path to G.

Search algorithms can be used to determine the sequence of actions.

---

# 15. Example: Utility-Based Navigation Agent

Suppose an agent has multiple possible routes.

It may consider:

- Distance.
- Time.
- Cost.
- Safety.

A utility function can combine these factors.

For example:

    utility =
        - distance
        - travel_time
        - cost
        + safety

The exact utility function depends on the application's objectives.

The agent selects the action sequence with the best expected utility.

---

# 16. Assignment Questions

### Question 1

Define an Artificial Agent.

### Question 2

Explain the difference between an agent and its environment.

### Question 3

Explain a Simple Reflex Agent with an example.

### Question 4

Explain a Model-Based Reflex Agent.

### Question 5

Explain a Goal-Based Agent.

### Question 6

Explain a Utility-Based Agent.

### Question 7

Compare the four major agent types.

### Question 8

Design an agent for a simple vacuum-cleaner environment.

### Question 9

Explain how search algorithms can be used by a Goal-Based Agent.

### Question 10

Explain why a Utility-Based Agent can be preferable when multiple valid solutions exist.

---

# 17. Learning Objectives

After completing this assignment, you should understand:

- What an artificial agent is.
- How agents interact with environments.
- What percepts and actions are.
- How Simple Reflex Agents work.
- How Model-Based Reflex Agents maintain internal state.
- How Goal-Based Agents use goals.
- How Utility-Based Agents compare outcomes.
- How search algorithms can support intelligent agents.

---

# 18. Key Takeaway

The progression can be remembered as:

Simple Reflex:

    "What do I perceive right now?"

Model-Based:

    "What do I perceive, and what do I remember?"

Goal-Based:

    "What action helps me reach my goal?"

Utility-Based:

    "Which available outcome is best?"
