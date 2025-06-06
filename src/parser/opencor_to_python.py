import re
import textwrap


def parse_opencor_to_python(opencor_code: str) -> str:
    # === Extract constants and their names ===
    const_names = re.findall(
        r'legend_constants\[\d+\] = "(.*?) in component', opencor_code
    )
    const_assignments = re.findall(r"constants\[(\d+)\] = ([\d\./]+)", opencor_code)
    constants = {}
    for idx, val in const_assignments:
        name = const_names[int(idx)] if int(idx) < len(const_names) else f"C{idx}"
        constants[name] = eval(val)

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
        r"def computeRates\(.*?\):(.+?)return\(rates\)", opencor_code, re.DOTALL
    )
    if not compute_rates_block:
        raise ValueError("Could not find computeRates function")
    rate_lines = compute_rates_block.group(1).split("\n")
    rate_exprs = [line.strip() for line in rate_lines if "rates[" in line]

    # === Build variable mappings ===
    var_map = {}
    for i, name in enumerate(state_names):
        var_map[f"states[{i}]"] = name
    for i, name in enumerate(const_names):
        var_map[f"constants[{i}]"] = name

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
        "import numpy as np",
        "from scipy.integrate import odeint",
        "",
        "# === True parameter values ===",
    ]
    for name, val in constants.items():
        py_lines.append(f"{name} = {val}")

    py_lines.append("\n# === ODE system ===")
    py_lines.append("def ODE(x, t):")
    unpack = ", ".join(states.keys())
    py_lines.append(f"    {unpack} = x")
    py_lines.append("    dxdt = [")
    for expr in rewritten_rates:
        py_lines.append(f"        {expr},")
    py_lines.append("    ]")
    py_lines.append("    return dxdt")

    py_lines.append("\n# === Initial condition ===")
    init_values = [states[k] for k in states]
    py_lines.append(f"x0 = {init_values}")

    return "\n".join(py_lines)


if __name__ == "__main__":
    with open("src/lorenz/generated/Lorenz_1963.py", "r") as f:
        opencor_code = f.read()

    code = parse_opencor_to_python(opencor_code)

    with open("synthetic_data_generator.py", "w") as f:
        f.write(code)
