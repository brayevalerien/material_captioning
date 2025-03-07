import os

from random import randint
from groq import Groq

from utils import *


def describe(
    b64_image: str, api_key: str, seed: int, model: str = "llama-3.2-11b-vision-preview"
) -> str:
    prompt = """Describe this image in full details, don't omit any information about it. Your caption must be at least 512 words long.
Ignore the background and the lighting, only describe the person in this portrait photograph.

Example result:
\"This is a portrait photograph of a woman in her 40s, she has long, straight blonde hair and blue eyes, her hair is tied in a ponny tails behind her back and flowing on her shoulders, [...]\"
"""
    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64_image}",
                        },
                    },
                ],
            }
        ],
        model=model,
        temperature=0.25,
        max_completion_tokens=1024,
        top_p=1,
        stream=False,
        seed=seed,
    )
    return completion.choices[0].message.content


def recaption(
    description: str,
    materials_replacement_rules: str,
    api_key: str,
    seed: int,
    model: str = "llama-3.3-70b-versatile",
) -> str:
    prompt = """You are an AI assistant tasked with transforming natural language descriptions of people, typically based on portrait photographs, by replacing certain elements with corresponding materials according to a predefined set of rules. Your goal is to reinterpret the description so the subject appears constructed from building materials, while preserving the original feel and texture of each element.

Guidelines:
1. Replacement Rules:
{REPLACEMENT_RULES}

2. Maintain Original Texture:
   - Replace elements with construction materials that mimic the original texture and feel. For example:
     - Soft materials (e.g., wool, silk) should be replaced by equally soft construction materials (e.g., mineral wool, fiberglass fabric).
     - Hard materials (e.g., metal, glass) should retain their hard, structured feel using analogous building materials (e.g., steel, concrete).

3. Flexible Adaptation:
   - If an element is not explicitly listed in the rules but fits the theme, adapt it using construction materials with matching textures. For example:
     - Replace \"long flowing hair\" with \"hair made of copper threads\" (soft and flowing).
     - Replace \"smooth leather shoes\" with \"smooth rubber shoes\" (soft and flexible).

4. Output:
   - Ensure the description is cohesive, vivid, and aligned with the theme of construction material textures.
   - Retain the original sentence structure wherever possible while applying replacements. Replace the original description, do not repeat it.
   - Don't add any other word or sentence like \"here is the rewritten prompt\" or \"let me know if you need another prompt to be converted\": your answer must be the new prompt and only the new prompt.
   - DO NOT add any element that it not mentioned in the original description, such as pair of glasses or a beard.
   - Insist on the material. Ensure the metal is shinny, the wood is grainy, the fabrics are soft or rough.
   - Go into greater details when you are describing hair and facial hair (beard/mustache).
   - You must highlight the specific materials and their texture, ensuring that they are clearly described in great details.
   - IGNORE the background and remove any mention of background elements.
   - keep the caption simple and ignore useless elements. Keep all the features discribing the subject. Remove contradictory elements.

Example:
User: \"The man had short dark brown hair, thin glasses, and wore light wool clothing.\"  
Assistant: \"The man had hair made out of dark brown brush, glasses made out of electrical wire, and wore mineral wool clothing.\"""".replace(
        "{REPLACEMENT_RULES}", materials_replacement_rules
    )
    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": description},
        ],
        model=model,
        temperature=0.25,
        max_completion_tokens=1024,
        top_p=1,
        stream=False,
        seed=seed,
    )
    return (
        completion.choices[0].message.content
        + "\n\nA studio shot with professional studio lighting. This image is a professional studio shot of a stop-motion animation character made of construction material"
    )


def write_caption(b64_image: str, api_key: str, seed: int = None) -> str:
    """
    Write a caption about an image with someone represented in the image, using construction materials, such as metalic wires for hair, insulating mineral wool for clothes, etc.

    Args:
        b64_image (str): a base 64 encoded image.
        api_key (str): the Groq API key.
        seed (str, optional): rng seed passed to the Groq API, leave to None to use a random seed. Defaults to None.

    Returns:
        str: Description of the character in the image, recaptioned using construction materials.
    """
    if seed is None:
        seed = randint(1 - 2**63, 2**63 - 1)
    description = describe(b64_image, api_key, seed)
    material_replacement_rules = load_materials()
    result = recaption(description, material_replacement_rules, api_key, seed)
    return result


if __name__ == "__main__":
    load_env()

    API_KEY = os.environ.get("GROQ_API_KEY")
    assert not API_KEY is None, "Groq API key not found in .env file."

    b64_image = load_b64_image("assets/selfie_guy.png")
    # b64_image = load_b64_image("assets/selfie_girl.png")

    description = describe(b64_image, API_KEY, 0)
    print("INITIAL DESCRIPTION")
    print(description)

    material_replacement_rules = load_materials()
    result = recaption(description, material_replacement_rules, API_KEY, 0)
    print("\n\n\nRE-WRITEN DESCRIPTION")
    print(result)

    print("\n\n\nDONE.")
