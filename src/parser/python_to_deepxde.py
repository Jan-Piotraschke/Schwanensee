import ast
import inspect
import textwrap
import re


def extract_parameters(source):
    """
    Extracts constant assignments like `self.constants[0] = 8.0/3.0`
    and maps them to C1, C2, C3, ...
    """
    matches = re.findall(r"self\.constants\[(\d+)\]\s*=\s*([^\n#]+)", source)
    return {f"C{int(i)+1}": expr.strip() for i, expr in matches}


def extract_initial_conditions(source):
    """
    Extracts x0 = [0.0, 1.0, 1.05] from the generated code.
    """
    match = re.search(r"x0\s*=\s*\[(.*?)\]", source)
    if match:
        return [float(x.strip()) for x in match.group(1).split(",")]
    return []


def convert_initial_conditions_class(ic_list):
    return [
        f"        self.ic{i + 1} = dde.icbc.IC(self.geom, lambda X: {val}, self.boundary, component={i})"
        for i, val in enumerate(ic_list)
    ]


def parse_ode_to_deepxde_method(ode_function, param_dict):
    source = textwrap.dedent(inspect.getsource(ode_function))
    function_ast = ast.parse(source).body[0]
    param_names = [arg.arg for arg in function_ast.args.args]
    state_var_name = param_names[0]
    state_vars = []
    body_lines = source.strip().split("\n")[1:]  # Skip def line

    # Extract the first line that unpacks multiple variables (e.g., x, y, z = x)
    unpacking_line = next(
        (line.strip() for line in body_lines if re.match(r"^\w+(, *\w+)+ *= *\w+", line.strip())),
        ""
    )
    if unpacking_line:
        lhs = unpacking_line.split("=")[0].strip()
        state_vars = [v.strip() for v in lhs.split(",")]
    else:
        raise ValueError("Could not find state unpacking line in ODE method.")

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
        "    def ODE_system(self, x, y, ex):",
        '        """Auto-generated DeepXDE system from ODE"""',
    ]
    for i, var in enumerate(state_vars):
        lines.append(f"        {var} = y[:, {i}:{i + 1}]")
    for i, var in enumerate(state_vars):
        lines.append(f"        d{var}_x = dde.grad.jacobian(y, x, i={i})")

    lines.append("        return [")
    for i, eq in enumerate(equations):
        for j, (pname, _) in enumerate(param_dict.items()):
            eq = eq.replace(pname, f"self.constants[{j}]")
        lines.append(f"            d{state_vars[i]}_x - ({eq}),")
    lines[-1] = lines[-1].rstrip(",")
    lines.append("        ]")
    return lines


def generate_deepxde_script(module):
    # Locate SyntheticDataGenerator class
    generator_cls = None
    for name, val in module.__dict__.items():
        if inspect.isclass(val) and name == "SyntheticDataGenerator":
            generator_cls = val
            break
    if not generator_cls:
        raise ValueError("SyntheticDataGenerator class not found in module")

    # Get ODE method
    try:
        ode_func = getattr(generator_cls, "ODE")
    except AttributeError:
        raise ValueError("ODE method not found in SyntheticDataGenerator class")

    # Full source of class for constant extraction
    class_source = inspect.getsource(generator_cls)
    param_dict = extract_parameters(class_source)
    x0_vals = extract_initial_conditions(class_source)

    ode_method_lines = parse_ode_to_deepxde_method(ode_func, param_dict)
    ic_lines = convert_initial_conditions_class(x0_vals)

    # Output script as class
    lines = [
        "# ==========================================",
        "# SECTION 2: PHYSICS MODEL DEFINITION",
        "#",
        "# This section defines the physical model",
        "# with unknown parameters that we're trying to identify,",
        "# including the system equations,",
        "# boundary conditions (BC),",
        "# and initial conditions (IC).",
        "# ==========================================",
        "import deepxde as dde",
        "import numpy as np",
        "",
        "class DeepXDESystem:",
        "    def __init__(self, maxtime):",
        f"        self.constants = [{', '.join('dde.Variable(1.0)' for _ in param_dict)}]",
        "        self.geom = dde.geometry.TimeDomain(0, maxtime)",
        "        self.boundary = lambda _, on_initial: on_initial",
    ] + ic_lines + [""] + ode_method_lines + [
        "",
        "    def get_observations(self, time, x):",
        "        observe_t, ob_y = time, x",
        "        return [",
        "            dde.icbc.PointSetBC(observe_t, ob_y[:, 0:1], component=0),",
        "            dde.icbc.PointSetBC(observe_t, ob_y[:, 1:2], component=1),",
        "            dde.icbc.PointSetBC(observe_t, ob_y[:, 2:3], component=2),",
        "        ]",
    ]

    return "\n".join(lines)
