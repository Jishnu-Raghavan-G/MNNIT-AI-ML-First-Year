from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Environment:
    """
    Simple two-room vacuum-cleaner environment.
    """
    rooms: Dict[str, str]
    agent_location: str = "A"

    def percept(self):
        """Return the agent's current percept."""
        return {
            "location": self.agent_location,
            "status": self.rooms[self.agent_location]
        }

    def execute(self, action):
        """Apply an action to the environment."""
        if action == "CLEAN":
            self.rooms[self.agent_location] = "clean"

        elif action == "MOVE_LEFT":
            if self.agent_location == "B":
                self.agent_location = "A"

        elif action == "MOVE_RIGHT":
            if self.agent_location == "A":
                self.agent_location = "B"


@dataclass
class SimpleReflexAgent:
    """
    Simple Reflex Agent.

    Decisions depend only on the current percept.
    """

    def choose_action(self, percept):
        if percept["status"] == "dirty":
            return "CLEAN"

        if percept["location"] == "A":
            return "MOVE_RIGHT"

        return "MOVE_LEFT"


@dataclass
class ModelBasedAgent:
    """
    Model-Based Reflex Agent.

    Maintains an internal representation of rooms.
    """

    internal_state: Dict[str, str] = field(
        default_factory=lambda: {
            "A": "unknown",
            "B": "unknown"
        }
    )

    def update_state(self, percept):
        location = percept["location"]
        status = percept["status"]

        self.internal_state[location] = status

    def choose_action(self, percept):
        self.update_state(percept)

        if percept["status"] == "dirty":
            return "CLEAN"

        other_room = (
            "B"
            if percept["location"] == "A"
            else "A"
        )

        if self.internal_state[other_room] != "clean":
            if percept["location"] == "A":
                return "MOVE_RIGHT"

            return "MOVE_LEFT"

        return "STOP"


@dataclass
class GoalBasedAgent:
    """
    Goal-Based Agent.

    Selects actions based on a desired destination.
    """

    goal: str

    def choose_action(self, current_location):
        if current_location == self.goal:
            return "STOP"

        if current_location == "A" and self.goal == "B":
            return "MOVE_RIGHT"

        if current_location == "B" and self.goal == "A":
            return "MOVE_LEFT"

        return "STOP"


@dataclass
class UtilityBasedAgent:
    """
    Utility-Based Agent.

    Chooses between possible outcomes using utility values.
    """

    utilities: Dict[str, float]

    def choose_best_option(self):
        if not self.utilities:
            return None

        return max(
            self.utilities,
            key=self.utilities.get
        )


def run_simple_reflex_demo():
    """Demonstrate a Simple Reflex Agent."""
    print("\n=== SIMPLE REFLEX AGENT ===")

    environment = Environment({
        "A": "dirty",
        "B": "dirty"
    })

    agent = SimpleReflexAgent()

    for step in range(6):
        percept = environment.percept()
        action = agent.choose_action(percept)

        print(
            f"Step {step + 1}: "
            f"location={percept['location']}, "
            f"status={percept['status']}, "
            f"action={action}"
        )

        if action == "STOP":
            break

        environment.execute(action)


def run_model_based_demo():
    """Demonstrate a Model-Based Agent."""
    print("\n=== MODEL-BASED AGENT ===")

    environment = Environment({
        "A": "dirty",
        "B": "dirty"
    })

    agent = ModelBasedAgent()

    for step in range(8):
        percept = environment.percept()
        action = agent.choose_action(percept)

        print(
            f"Step {step + 1}: "
            f"location={percept['location']}, "
            f"status={percept['status']}, "
            f"internal_state={agent.internal_state}, "
            f"action={action}"
        )

        if action == "STOP":
            break

        environment.execute(action)


def run_goal_based_demo():
    """Demonstrate a Goal-Based Agent."""
    print("\n=== GOAL-BASED AGENT ===")

    agent = GoalBasedAgent(goal="B")

    current_location = "A"

    for step in range(3):
        action = agent.choose_action(
            current_location
        )

        print(
            f"Step {step + 1}: "
            f"location={current_location}, "
            f"goal={agent.goal}, "
            f"action={action}"
        )

        if action == "STOP":
            break

        if action == "MOVE_RIGHT":
            current_location = "B"

        elif action == "MOVE_LEFT":
            current_location = "A"


def run_utility_based_demo():
    """Demonstrate utility-based decision making."""
    print("\n=== UTILITY-BASED AGENT ===")

    utilities = {
        "Route A": 70,
        "Route B": 60,
        "Route C": 85
    }

    agent = UtilityBasedAgent(
        utilities=utilities
    )

    print("Available options:")

    for route, utility in utilities.items():
        print(
            f"{route}: utility={utility}"
        )

    best_option = agent.choose_best_option()

    print(
        f"\nSelected option: {best_option}"
    )


def print_agent_comparison():
    """Print a compact comparison of agent types."""
    print("\n=== AGENT COMPARISON ===")

    comparison = [
        (
            "Simple Reflex",
            "Current percept",
            "No substantial state"
        ),
        (
            "Model-Based",
            "Percept + internal state",
            "Yes"
        ),
        (
            "Goal-Based",
            "Goal achievement",
            "Often"
        ),
        (
            "Utility-Based",
            "Best outcome",
            "Yes"
        )
    ]

    for name, decision_basis, memory in comparison:
        print(f"\n{name}")
        print(f"Decision basis: {decision_basis}")
        print(f"Internal state: {memory}")


def main():
    print("ARTIFICIAL AGENTS")

    run_simple_reflex_demo()
    run_model_based_demo()
    run_goal_based_demo()
    run_utility_based_demo()
    print_agent_comparison()


if __name__ == "__main__":
    main()
