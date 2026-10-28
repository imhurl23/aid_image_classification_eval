"""Render cached independent model classifications as native Braintrust messages."""

MODELS = ("luna", "jev", "clef", "d1", "d1-ultra")
NAMES = {"luna": "Luna", "jev": "Jev", "clef": "Clef", "d1": "D1", "d1-ultra": "D1 Ultra"}


def review_input(image):
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What is the overall scene category in this aerial image?",
                },
                {"type": "image_url", "image_url": {"url": image}},
            ],
        }
    ]


def review_output(results):
    # A display of independent cached decisions, not a conversation among models.
    return [
        {
            "role": "assistant",
            "name": m,
            "content": NAMES[m]
            + ": "
            + (
                results[m]["label"]
                if results[m]["status"] == "ok"
                else "[" + results[m]["status"] + ": no category returned]"
            ),
        }
        for m in MODELS
    ]


def review_image(input):
    if isinstance(input, dict):
        return input["image"]
    return input[0]["content"][1]["image_url"]["url"]
