import ast
import inspect
import textwrap
import re


def extract_parameters(source):
    param_lines = [
        line.strip()
        for line in source.splitlines()
        if re.match(r"^\w+\s*=\s*[\d.]+$", line)
    ]
    params = {}
    for line in param_lines:
        name, value = map(str.strip, line.split("="))
        params[name] = float(value)
    return params


def extract_initial_conditions(source):
    match = re.search(r"x0\s*=\s*\[(.*?)\]", source)
    if match:
        return [float(x.strip()) for x in match.group(1).split(",")]
    return []


def convert_parameters_to_variables(param_dict):
    return [f"{k} = dde.Variable(1.0)" for k in param_dict]


def convert_initial_conditions(x0_list):
    ic_lines = []
    for i, val in enumerate(x0_list):
        ic_lines.append(
            f"ic{i + 1} = dde.icbc.IC(geom, lambda X: {val}, boundary, component={i})"
        )
    return ic_lines


def parse_ode_to_deepxde(ode_function, param_dict):
    source = textwrap.dedent(inspect.getsource(ode_function))
    function_ast = ast.parse(source).body[0]
    param_names = [arg.arg for arg in function_ast.args.args]
    state_var_name = param_names[0]
    state_vars = []
    body_lines = source.strip().split("\n")[1:]  # Skip def line

    unpacking_line = next(
        (line.strip() for line in body_lines if "=" in line and state_var_name in line),
        "",
    )
    if unpacking_line:
        vars_str = unpacking_line.split("=")[0].strip().strip("[] ")
        state_vars = [v.strip() for v in vars_str.split(",")]

    return_var = None
    for line in body_lines:
        if "return" in line:
            match = re.search(r"return\s+(\w+)", line.strip())
            if match:
                return_var = match.group(1)
                break

    equations = []
    if return_var:
        collecting = False
        assignment_lines = []
        for line in body_lines:
            stripped = line.strip()
            if stripped.startswith(f"{return_var} = ["):
                collecting = True
                assignment_lines.append(stripped.split("=", 1)[1].strip())
            elif collecting:
                assignment_lines.append(stripped)
                if stripped.endswith("]"):
                    break

        code_block = " ".join(assignment_lines)
        code_block = code_block.strip()[1:-1]
        equations = [eq.strip() for eq in code_block.split(",") if eq.strip()]

    lines = [
        f"def ODE_system(x, y, ex):",
        '    """Auto-generated DeepXDE system from ODE"""',
    ]
    for i, var in enumerate(state_vars):
        lines.append(f"    {var} = y[:, {i}:{i + 1}]")
    for i, var in enumerate(state_vars):
        lines.append(f"    d{var}_x = dde.grad.jacobian(y, x, i={i})")
    lines.append("    return [")
    for i, eq in enumerate(equations):
        eq = eq.replace("**", "^")  # Optional formatting cleanup
        for p in param_dict.keys():
            eq = eq.replace(
                p, f"C{i + 1}"
            )  # Replace with generic DeepXDE variable name
        lines.append(f"        d{state_vars[i]}_x - ({eq}),")
    lines[-1] = lines[-1].rstrip(",")
    lines.append("    ]")
    return "\n".join(lines)


def generate_deepxde_script(module):
    source = inspect.getsource(module)
    param_dict = extract_parameters(source)
    param_vars = convert_parameters_to_variables(param_dict)
    x0_vals = extract_initial_conditions(source)
    ic_lines = convert_initial_conditions(x0_vals)

    # Get the actual ODE function from the module
    ode_func = None
    for name, val in module.__dict__.items():
        if callable(val) and name == "ODE":
            ode_func = val
            break

    if not ode_func:
        raise ValueError("ODE function not found in module")

    ode_block = parse_ode_to_deepxde(ode_func, param_dict)

    lines = (
        [
            "# ==========================================",
            "# SECTION 2: PHYSICS MODEL DEFINITION",
            "#",
            "# This section defines the physical model",
            "# with unknown parameters that we're trying to identify,",
            "# including the system equations,",
            "# boundary conditions (BC),",
            "# and initial conditions (IC).",
            "# ==========================================",
            "# parameters to be identified",
        ]
        + param_vars
        + [
            "",
            "# define system ODEs",
            ode_block,
            "",
            "def boundary(_, on_initial):",
            "    return on_initial",
            "",
            "# define time domain",
            "geom = dde.geometry.TimeDomain(0, maxtime)",
            "",
            "# Initial conditions",
        ]
        + ic_lines
        + [
            "",
            "# Get the training data",
            "observe_t, ob_y = time, x",
            "observe_y0 = dde.icbc.PointSetBC(observe_t, ob_y[:, 0:1], component=0)",
            "observe_y1 = dde.icbc.PointSetBC(observe_t, ob_y[:, 1:2], component=1)",
            "observe_y2 = dde.icbc.PointSetBC(observe_t, ob_y[:, 2:3], component=2)",
        ]
    )
    return "\n".join(lines)


# def test_parser():
#     import synthetic_data_generator as sdg
#     deepxde_code = generate_deepxde_script(sdg)

#     with open("physio_sensai_model.py", "w") as f:
#         f.write(deepxde_code)

# # Run the test
# test_parser()
