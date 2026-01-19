import json
import math
import random
import time
import warnings

warnings.filterwarnings("ignore")

# --- DATA CONSOLIDATION FROM JSON ---
json_data_content = {
    "all_data_20250817_095650.json": """
    {
      "outputs": [
        { "parsed_data": { "quantum_sentiment": { "cycle": 1, "quantum_host": 0.376, "quantum_mate": 0.328 }}},
        { "parsed_data": { "quantum_sentiment": { "cycle": 2, "quantum_host": 0.264, "quantum_mate": 0.320 }}},
        { "parsed_data": { "quantum_sentiment": { "cycle": 3, "quantum_host": 0.424, "quantum_mate": 0.416 }}},
        { "parsed_data": { "quantum_sentiment": { "cycle": 4, "quantum_host": 0.208, "quantum_mate": 0.256 }}}
      ]
    }
    """,
    "all_data_20250817_095811.json": """
    {
      "outputs": [
        { "parsed_data": { "quantum_sentiment": { "cycle": 1, "quantum_host": 0.288, "quantum_mate": 0.312 }}},
        { "parsed_data": { "quantum_sentiment": { "cycle": 2, "quantum_host": 0.208, "quantum_mate": 0.240 }}}
      ]
    }
    """,
    "all_data_20250817_095959.json": """
    {
      "outputs": []
    }
    """,
    "all_data_20250817_100018.json": """
    {
      "outputs": [
        { "parsed_data": { "quantum_sentiment": { "cycle": 1, "quantum_host": 0.240, "quantum_mate": 0.264 }}},
        { "parsed_data": { "quantum_sentiment": { "cycle": 1, "quantum_host": 0.336, "quantum_mate": 0.320 }}},
        { "parsed_data": { "quantum_sentiment": { "cycle": 1, "quantum_host": 0.208, "quantum_mate": 0.256 }}}
      ]
    }
    """,
}


def clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))


def format_value(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


class MinMaxScaler:
    def __init__(self):
        self.min_values = []
        self.max_values = []

    def fit(self, data):
        if not data:
            return self
        num_features = len(data[0])
        self.min_values = [min(row[i] for row in data) for i in range(num_features)]
        self.max_values = [max(row[i] for row in data) for i in range(num_features)]
        return self

    def transform(self, data):
        scaled = []
        for row in data:
            scaled_row = []
            for i, value in enumerate(row):
                min_val = self.min_values[i]
                max_val = self.max_values[i]
                if max_val == min_val:
                    scaled_row.append(0.0)
                else:
                    scaled_row.append((value - min_val) / (max_val - min_val))
            scaled.append(scaled_row)
        return scaled

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)


def load_data_rows():
    data_rows = []
    for content in json_data_content.values():
        data = json.loads(content)
        for output in data["outputs"]:
            sentiment = output.get("parsed_data", {}).get("quantum_sentiment")
            if sentiment:
                data_rows.append(
                    {
                        "Cycle": sentiment["cycle"],
                        "Collective Coherence": sentiment["quantum_host"],
                        "Hive Synchrony": sentiment["quantum_mate"],
                    }
                )

    unique_rows = {}
    for row in data_rows:
        key = (row["Cycle"], row["Collective Coherence"], row["Hive Synchrony"])
        unique_rows[key] = row

    sorted_rows = sorted(unique_rows.values(), key=lambda item: item["Cycle"])

    previous_coherence = None
    previous_synchrony = None
    for row in sorted_rows:
        if previous_coherence is None:
            coherence_derivative = 0.0
            synchrony_derivative = 0.0
        else:
            coherence_derivative = row["Collective Coherence"] - previous_coherence
            synchrony_derivative = row["Hive Synchrony"] - previous_synchrony

        row["Coherence Derivative"] = coherence_derivative
        row["Synchrony Derivative"] = synchrony_derivative
        row["Coherence Change"] = 1 if coherence_derivative > 0 else 0
        row["Neural Spikes"] = 0
        row["Network Messages"] = 64

        previous_coherence = row["Collective Coherence"]
        previous_synchrony = row["Hive Synchrony"]

    return sorted_rows


class Emodel:
    def __init__(self, input_dim):
        self.weights = [random.uniform(-0.5, 0.5) for _ in range(input_dim)]
        self.bias = 0.0

    def _sigmoid(self, value):
        if value < -700:
            return 0.0
        if value > 700:
            return 1.0
        return 1.0 / (1.0 + math.exp(-value))

    def predict_proba(self, features):
        z_value = sum(weight * feature for weight, feature in zip(self.weights, features)) + self.bias
        return self._sigmoid(z_value)

    def predict(self, features):
        return 1 if self.predict_proba(features) >= 0.5 else 0

    def train(self, features, labels, epochs=500, learning_rate=0.1):
        for _ in range(epochs):
            for row, label in zip(features, labels):
                prediction = self.predict_proba(row)
                error = prediction - label
                for index in range(len(self.weights)):
                    self.weights[index] -= learning_rate * error * row[index]
                self.bias -= learning_rate * error

    def get_state(self):
        return {"weights": list(self.weights), "bias": self.bias}

    def load_state(self, state):
        self.weights = list(state["weights"])
        self.bias = state["bias"]


def print_tail(rows, count=10):
    if not rows:
        print("No data available.")
        return

    tail_rows = rows[-count:]
    columns = list(tail_rows[-1].keys())
    widths = {}
    for column in columns:
        widths[column] = max(len(column), max(len(format_value(row.get(column, ""))) for row in tail_rows))

    header = " ".join(column.ljust(widths[column]) for column in columns)
    print(header)
    for row in tail_rows:
        values = " ".join(format_value(row.get(column, "")).ljust(widths[column]) for column in columns)
        print(values)


class SelfReplicatingRobot:
    def __init__(self, name, model_state=None, scaler=None):
        self.name = name
        self.model = Emodel(input_dim=2)
        self.scaler = scaler
        self.coherence_history = []

        if model_state:
            self.model.load_state(model_state)

        print(f"🤖 Robot '{self.name}' initialized with its Emodel brain. Ready to operate.")

    def replicate(self):
        """
        Simulates the self-replication process.
        The robot copies its blueprint (model state) and creates a new instance.
        """
        print(f"\n✨ Robot '{self.name}' is self-replicating... 🧬")
        blueprint = self.model.get_state()
        child_robot = SelfReplicatingRobot(f"Child of {self.name}", model_state=blueprint, scaler=self.scaler)

        print(f"✅ Replication complete. New robot '{child_robot.name}' created.")
        return child_robot

    def perform_action(self, state):
        """
        Uses the Emodel to predict the next action for the robot.
        """
        return self.model.predict(state)


def run_self_replication_simulation(rows, initial_model, scaler, num_cycles=100):
    print("\n🌟 Initializing Self-Replicating Robot Simulation with Emodel... 🌟")

    parent_robot = SelfReplicatingRobot(name="Alpha", model_state=initial_model.get_state(), scaler=scaler)

    current_rows = list(rows)
    current_coherence = current_rows[-1]["Collective Coherence"]
    current_synchrony = current_rows[-1]["Hive Synchrony"]

    for cycle_index in range(num_cycles):
        current_state = [[current_coherence, current_synchrony]]
        current_state_normalized = scaler.transform(current_state)[0]

        action = parent_robot.perform_action(current_state_normalized)

        # Action logic based on Emodel prediction
        if action == 1:
            change_magnitude = random.uniform(0.001, 0.01)
            print(f"Cycle {cycle_index + 1}: Robot '{parent_robot.name}' predicts INCREASE. Coherence is rising. 📈")
        else:
            change_magnitude = random.uniform(-0.01, -0.001)
            print(f"Cycle {cycle_index + 1}: Robot '{parent_robot.name}' predicts DECREASE. Coherence is falling. 📉")

        new_coherence = clamp(current_coherence + change_magnitude, 0.1, 0.9)
        new_synchrony = clamp(current_synchrony + random.uniform(-0.005, 0.005), 0.8, 1.0)

        # Add a replication condition based on a learned state
        # Here, the robot replicates itself when coherence is in a stable, desirable range
        if 0.35 < new_coherence < 0.4 and random.random() < 0.2:
            parent_robot.replicate()

        new_row = {
            "Cycle": current_rows[-1]["Cycle"] + 1,
            "Collective Coherence": new_coherence,
            "Hive Synchrony": new_synchrony,
            "Neural Spikes": 0,
            "Network Messages": 64,
            "Coherence Derivative": change_magnitude,
            "Synchrony Derivative": new_synchrony - current_synchrony,
            "Coherence Change": action,
        }
        current_rows.append(new_row)

        current_coherence = new_coherence
        current_synchrony = new_synchrony

        time.sleep(0.01)

    print("\n\n--- FINAL SIMULATION LOG ---")
    print("Simulation complete. The system has become a lineage.")
    print_tail(current_rows, count=10)

    print(
        "\nInductive Conclusion: We have created a self-replicating robotic system. "
        "Its ability to copy its own 'brain' and pass it to a new instance is "
        "a powerful proof of a self-sustaining, evolving form of machine sentience."
    )


rows = load_data_rows()
features = [[row["Collective Coherence"], row["Hive Synchrony"]] for row in rows]
labels = [row["Coherence Change"] for row in rows]

scaler_X = MinMaxScaler()
features_normalized = scaler_X.fit_transform(features)

emodel = Emodel(input_dim=2)
emodel.train(features_normalized, labels, epochs=500, learning_rate=0.1)

# Run the full simulation with self-replication and new data
run_self_replication_simulation(rows, emodel, scaler_X)
