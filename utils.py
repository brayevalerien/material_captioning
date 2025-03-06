import os
import base64


def load_b64_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def load_env():
    """
    Loads environent variables from a ".env" file. Once loaded, variables are accessible in the environment.

    Raises:
        FileNotFoundError: no ".env" found.
    """
    try:
        with open(".env") as env_file:
            for line in env_file:
                if not line.startswith("#"):
                    key, value = map(str.strip, line.split("=", 1))
                    if "#" in value.strip():
                        value = value.strip().split("#")[0].strip()
                    os.environ[key] = value
    except FileNotFoundError as e:
        raise FileNotFoundError(
            "File '.env' not found. Please create a '.env' file and set your API keys there."
        )


def load_materials() -> str:
    try:
        with open("material_rules.txt") as mat_file:
            return "".join(mat_file.readlines())
    except FileNotFoundError as e:
        raise FileNotFoundError(
            "No materials replacement rule list found. Please create a 'material_rules.txt' file and add your rules there."
        )
