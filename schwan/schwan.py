import argparse
import os
import importlib.util
import sys

from parser.opencor_to_python import parse_opencor_to_python
from parser.python_to_deepxde import generate_deepxde_script


def main():
    parser = argparse.ArgumentParser(
        description="Schwan: A parser tool for OpenCOR code."
    )
    parser.add_argument(
        "--python", action="store_true", help="Convert OpenCOR code to pure Python ODE"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input OpenCOR-generated Python file",
    )

    args = parser.parse_args()

    if args.python:
        input_path = args.input

        if not os.path.isfile(input_path):
            print(f"Input file '{input_path}' not found.")
            return

        with open(input_path, "r") as infile:
            opencor_code = infile.read()

        try:
            converted_code = parse_opencor_to_python(opencor_code)
        except Exception as e:
            print(f"Error during parsing: {e}")
            return

        # Determine output path in the same directory
        output_dir = os.path.dirname(input_path)
        output_path = os.path.join(output_dir, "synthetic_data_generator.py")

        with open(output_path, "w") as outfile:
            outfile.write(converted_code)

        print(f"Conversion successful! Output saved to: {output_path}")

        # Dynamically import the synthetic_data_generator module
        spec = importlib.util.spec_from_file_location(
            "synthetic_data_generator", output_path
        )
        sdg = importlib.util.module_from_spec(spec)
        sys.modules["synthetic_data_generator"] = sdg
        try:
            spec.loader.exec_module(sdg)
        except Exception as e:
            print(f"Error importing synthetic_data_generator: {e}")
            return

        # Generate DeepXDE code
        try:
            deepxde_code = generate_deepxde_script(sdg)
        except Exception as e:
            print(f"Error generating DeepXDE code: {e}")
            return

        dde_output_path = os.path.join(output_dir, "physio_sensai_model.py")
        with open(dde_output_path, "w") as f:
            f.write(deepxde_code)

        print(f"DeepXDE model script saved to: {dde_output_path}")

    else:
        print("No conversion flag set.")


if __name__ == "__main__":
    main()
