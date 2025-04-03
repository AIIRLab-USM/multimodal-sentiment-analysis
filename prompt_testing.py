from caption_generation import CaptionGenerator
import os

os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = pow(2, 40).__str__()

IN_PATH = f"data{os.path.sep}processed_data{os.path.sep}custom_artpedia.csv"

OUT_PATHS = [
        f"data{os.path.sep}caption_data{os.path.sep}prompt_eng{os.path.sep}zero-shot_artpedia-captions.csv",
        f"data{os.path.sep}caption_data{os.path.sep}prompt_eng{os.path.sep}one-shot_artpedia-captions.csv",
        f"data{os.path.sep}caption_data{os.path.sep}prompt_eng{os.path.sep}few-shot_artpedia-captions.csv",
        f"data{os.path.sep}caption_data{os.path.sep}prompt_eng{os.path.sep}multimodal-cot_artpedia-captions.csv",
        f"data{os.path.sep}caption_data{os.path.sep}prompt_eng{os.path.sep}self-consistency_artpedia-captions.csv"
    ]

PROMPTS = [
        # Zero-shot
        "USER: <image>\n"
        "Caption this painting\n"
        "ASSISTANT:",

        # One-shot
        "USER:\n"
        "Example:\n"
        "The result was a sparse landscape in which six old trees stand on a small bank in the lower foreground, across an expanse of water from some distant hills in the background.\n\n"
        "Now, caption this painting:\n"
        "<image>\n",

        # Few-shot
        "USER:\n"
        "Example 1:\n"
        "The painting depicts the Madonna with Child crowned by two flying angels, sitting inside a rose garden in typical late Gothic style.\n\n"
        "Example 2:\n"
        "Now in the Prado, Madrid, it depicts a seated and serene Virgin Mary dressed in a long, flowing red robe lined with gold-coloured thread.\n\n"
        "Example 3:\n"
        "The panel shows the Virgin in a domestic interior, two attendant angels, the archangel Gabriel dressed in ecclesiastical robes, and a hovering dove representing the Holy Spirit.\n\n"
        "Now, caption this painting:\n"
        "<image>\n",

        # Multimodal Chain-of-Thought
        "USER: <image>\n"
        "Carefully consider the main objects or people in the painting.\n"
        "Thoroughly think about their positions, relative to each other.\n"
        "Observe their shapes, colors, and any other distinguishing features.\n"
        "Note any patterns, structures, or significant visual elements.\n"
        "Now, caption the painting.\n"
        "ASSISTANT:",

        # Self-Consistency
        "USER: <image>\n"
        "Consider the objects, people, and key details from different viewpoints.\n"
        "Internally generate and consider various different captions for the painting.\n"
        "Lastly, provide only the most accurate and descriptive caption.\n"
        "ASSISTANT:"

    ]

def generate_test_captions():
    cap_gen = CaptionGenerator()
    cap_gen.setup()

    for out_path, prompt in zip(OUT_PATHS, PROMPTS):
        cap_gen.from_csv(IN_PATH, out_path, prompt)


if __name__ == "__main__":
        generate_test_captions()