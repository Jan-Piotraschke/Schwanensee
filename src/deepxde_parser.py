import inspect
import ast
import re
import textwrap

def parse_ode_to_deepxde(ode_function):
    """
    Parse a Python ODE function to DeepXDE format.

    Args:
        ode_function: The ODE function to parse

    Returns:
        str: DeepXDE function code
    """
    # Get the source code of the function
    source = textwrap.dedent(inspect.getsource(ode_function))

    # Parse the function to get parameter names
    function_ast = ast.parse(source).body[0]
    param_names = [arg.arg for arg in function_ast.args.args]

    # Extract state variable name and time variable name
    state_var_name = param_names[0]
    time_var_name = param_names[1] if len(param_names) > 1 else 't'

    # Find unpacking of state variables
    unpacking_line = None
    body_lines = source.strip().split('\n')[1:]  # Skip the function definition
    for line in body_lines:
        if '=' in line and state_var_name in line.split('=')[1]:
            unpacking_line = line.strip()
            break

    # Extract state variable components
    state_vars = []
    if unpacking_line:
        match = re.search(r'([^=]+)=\s*' + state_var_name, unpacking_line)
        if match:
            vars_str = match.group(1).strip()
            # Remove brackets and split by comma
            vars_str = vars_str.strip('[] ')
            state_vars = [v.strip() for v in vars_str.split(',')]

    # Find the return statement to extract equations
       # Step 1: Find the return variable name (e.g., dxdt)
    return_var = None
    for line in body_lines:
        stripped = line.strip()
        if stripped.startswith("return"):
            match = re.match(r"return\s+(\w+)", stripped)
            if match:
                return_var = match.group(1)
                break

    # Step 2: Find where the return_var is assigned (e.g., dxdt = [ ... ])
    equations = []
    if return_var:
        collecting = False
        assignment_lines = []
        for line in body_lines:
            stripped = line.strip()
            if stripped.startswith(f"{return_var} = ["):
                collecting = True
                assignment_lines.append(stripped.split("=", 1)[1].strip())  # Keep RHS
            elif collecting:
                assignment_lines.append(stripped)
                if stripped.endswith("]"):
                    break

        # Step 3: Clean and split expressions
        if assignment_lines:
            code_block = " ".join(assignment_lines)
            code_block = code_block.strip()[1:-1]  # remove surrounding [ ]
            raw_eqs = code_block.split(',')
            equations = [eq.strip() for eq in raw_eqs if eq.strip()]



    # Generate DeepXDE function
    function_name = ode_function.__name__
    capitalized_name = function_name[0].upper() + function_name[1:] if function_name else "System"

    deepxde_code = [
        f"def {capitalized_name}_system(x, y, ex):",
        f'    """Auto-generated DeepXDE system from {function_name}"""'
    ]

    # Add state variable unpacking
    for i, var in enumerate(state_vars):
        deepxde_code.append(f"    {var} = y[:, {i}:{i+1}]")

    # Add derivatives
    for i, var in enumerate(state_vars):
        deepxde_code.append(f"    d{var}_x = dde.grad.jacobian(y, x, i={i})")

    # Process equations for DeepXDE format
    deepxde_eqs = []
    for i, eq in enumerate(equations):
        # Replace any time-dependent function with 'ex'
        eq = re.sub(r'ex_func`$$[^)$$`]+\)', 'ex', eq)
        # Create DeepXDE equation
        deepxde_eq = f"d{state_vars[i]}_x - ({eq})"
        deepxde_eqs.append(deepxde_eq)

    # Add return statement with equations
    deepxde_code.append("    return [")
    for eq in deepxde_eqs:
        deepxde_code.append(f"        {eq},")
    deepxde_code[-1] = deepxde_code[-1].rstrip(',')  # Remove trailing comma
    deepxde_code.append("    ]")

    return "\n".join(deepxde_code)


def test_parser():
    # Define the original Lorenz model
    def LorezODE(x, t):
        x1, x2, x3 = x
        dxdt = [
            C1true * (x2 - x1),
            x1 * (C2true - x3) - x2,
            x1 * x2 - C3true * x3 + ex_func(t),
        ]
        return dxdt

    # Generate DeepXDE code
    deepxde_code = parse_ode_to_deepxde(LorezODE)
    print(deepxde_code)

# Run the test
test_parser()
