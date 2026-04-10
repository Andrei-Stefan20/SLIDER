"""Names SAE features by querying an LLM with visual descriptions."""

from typing import List, Optional

from openai import OpenAI


class LLMFeatureNamer:
    """Calls the OpenAI API to assign a human-readable name to an SAE feature.

    Given CLIP-generated descriptions of the images that most strongly
    activate (or suppress) a feature, the LLM infers the underlying visual
    property that varies.
    """

    SYSTEM_PROMPT = (
        "You are a computer vision expert specialising in interpreting "
        "latent features of neural networks.  Your task is to name the "
        "visual property captured by a specific feature, based on "
        "descriptions of images that produce high vs low activations."
    )

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
    ) -> None:
        """Initialise the LLM namer.

        Args:
            model: OpenAI model identifier.
            api_key: OpenAI API key.  If ``None``, the client will read
                ``OPENAI_API_KEY`` from the environment.
        """
        self.model = model
        self.client = OpenAI(api_key=api_key)

    def name_feature(
        self,
        top_descriptions: List[str],
        bottom_descriptions: List[str],
    ) -> str:
        """Produce a concise name for the visual property of an SAE feature.

        Args:
            top_descriptions: CLIP descriptions of images with *high*
                activation on the feature.
            bottom_descriptions: CLIP descriptions of images with *low*
                activation on the feature.

        Returns:
            A 2–4 word slider label, e.g. ``"leaf margin complexity"``.
        """
        top_str = "; ".join(top_descriptions)
        bottom_str = "; ".join(bottom_descriptions)

        user_message = (
            f"These images have HIGH activation: {top_str}.\n"
            f"These images have LOW activation: {bottom_str}.\n"
            "In 2-4 words, what visual property varies between them?"
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            max_tokens=32,
            temperature=0.2,
        )

        return response.choices[0].message.content.strip()

    def name_features_batch(
        self,
        feature_descriptions: list[dict],
    ) -> list[str]:
        """Name multiple features sequentially.

        Args:
            feature_descriptions: List of dicts, each with keys
                ``"top_descriptions"`` and ``"bottom_descriptions"``.

        Returns:
            List of feature name strings in the same order.
        """
        return [
            self.name_feature(
                fd["top_descriptions"],
                fd["bottom_descriptions"],
            )
            for fd in feature_descriptions
        ]
