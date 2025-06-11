import re

def parse_opencor_to_python(opencor_code: str) -> str:
    # === Extract constants and their names ===
    const_names = re.findall(
        r'legend_constants\[\d+\] = "(.*?) in component', opencor_code
    )
    const_assignments = re.findall(r"constants\[(\d+)\] = ([\d\./]+)", opencor_code)
    constants = {}
    for idx, val in const_assignments:
        name = const_names[int(idx)] if int(idx) < len(const_names) else f"C{idx}"
        constants[name] = val  # keep as string for reuse in initConsts

    # === Extract state initial values and their names ===
    state_names = re.findall(
        r'legend_states\[\d+\] = "(.*?) in component', opencor_code
    )
    state_assignments = re.findall(r"states\[(\d+)\] = ([-\d.eE]+)", opencor_code)
    states = {}
    for idx, val in state_assignments:
        name = state_names[int(idx)] if int(idx) < len(state_names) else f"x{idx + 1}"
        states[name] = float(val)

    # === Parse the computeRates function ===
    compute_rates_block = re.search(
        r"def computeRates.*?:\s*(.*?)\s*return\(rates\)", opencor_code, re.DOTALL
    )
    if not compute_rates_block:
        raise ValueError("Could not find computeRates function")
    rate_lines = compute_rates_block.group(1).split("\n")
    rate_exprs = [line.strip() for line in rate_lines if "rates[" in line]

    # === Build variable mappings ===
    var_map = {}
    for i, name in enumerate(state_names):
        var_map[f"states[{i}]"] = name.split(" ")[0]
    for i, name in enumerate(const_names):
        var_map[f"constants[{i}]"] = f"self.constants[{i}]"

    # === Rewrite rate expressions ===
    rewritten_rates = []
    for line in rate_exprs:
        match = re.match(r"rates\[(\d+)\] = (.+)", line)
        if not match:
            continue
        expr = match.group(2)
        for k, v in var_map.items():
            expr = expr.replace(k, v)
        rewritten_rates.append(expr.strip())

    # === Assemble output Python code ===
    py_lines = [
        "# === Auto-generated file. Do not modify! ===\n",
        "# ==========================================",
        "# SECTION 1: DATA GENERATION & INPUT PREPARATION",
        "#",
        "# Simulated Unobservable Data:",
        "# This section covers creating the synthetic data",
        "# from the Lorenz system using the true parameters",
        "# and preparing the input data.",
        "# ==========================================",
        "import numpy as np",
        "",
        "class SyntheticDataGenerator:",
        "    sizeAlgebraic = 0",
        f"    sizeStates = {len(state_names)}",
        f"    sizeConstants = {len(const_names)}",
        "",
        "    def __init__(self):",
        "        self.constants = [0.0] * SyntheticDataGenerator.sizeConstants",
        "",
        "    def initConsts(self):",
        "        states = [0.0] * SyntheticDataGenerator.sizeStates",
    ]

    for i, val in enumerate(states.values()):
        py_lines.append(f"        states[{i}] = {val}")
    for i, (name, val) in enumerate(constants.items()):
        py_lines.append(f"        self.constants[{i}] = {val}")
    py_lines.append("        return (states, self.constants)")
    py_lines.append("")
    py_lines.append("    def ODE(self, x, t):")
    unpack = ", ".join([name.split(" ")[0] for name in state_names])
    py_lines.append(f"        {unpack} = x")
    py_lines.append("        dxdt = [")
    for expr in rewritten_rates:
        py_lines.append(f"            {expr},")
    py_lines.append("        ]")
    py_lines.append("        return dxdt")

    return "\n".join(py_lines)


if __name__ == "__main__":
    with open("src/lorenz/generated/Lorenz_1963.py", "r") as f:
        opencor_code = f.read()

    code = parse_opencor_to_python(opencor_code)

    with open("synthetic_data_generator.py", "w") as f:
        f.write(code)
