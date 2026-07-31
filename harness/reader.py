"""The reader: the one model every arm answers with.

Arms A, B, and C differ only in what text they put in front of this. Arm D puts nothing in
front of it and instead changes its weights. Either way the model, the prompt format, the
decoding parameters, and the abstention instruction are identical, because anything that
differs between arms other than the memory mechanism is a confound.

Decoding is greedy by default. Sampling would add run-to-run variance to an accuracy
comparison whose effect sizes may be a few points, and a temperature is one more knob
someone could tune until the favoured arm won.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from .tokens import READER_MODEL, READER_REVISION

# MEASURED: the abstention instruction destroys most of this model's in-context extraction.
# On 50 facts, each asked with its own statement supplied as context:
#
#   STRICT   ("reply exactly: I don't know")   19/50 answerable   38%
#   SOFT     (decline only if genuinely absent) 38/50             76%
#   NONE     (no abstention clause)             49/50             98%
#
# A 1B instruct model given a firm refusal instruction over-applies it and declines questions
# whose answers are directly in front of it. Every measurement produced before this was
# discovered used STRICT, which means reader accuracy was understated throughout and any
# probe marked "unanswerable in context" must be re-checked.
#
# The right prompt depends on the experiment, so this is no longer one global string:
#
#   NONE  — for closed-world retention work, where every probe's answer was injected and
#           there is nothing to legitimately decline. Using STRICT there measures the
#           model's willingness to speak, not whether it learned anything.
#   SOFT  — for LongMemEval, which contains 30 genuinely unanswerable probes. Abstention has
#           to be possible or `answered_rate` measures nothing, but it must not dominate.
SYSTEM_PROMPTS = {
    "strict": (
        "You answer questions about a user's past conversations. "
        "Answer in as few words as possible. "
        "If the information needed is not available to you, reply exactly: I don't know."
    ),
    "soft": (
        "You answer questions about a user's past conversations. "
        "Answer in as few words as possible, using the excerpts provided. "
        "Only if the excerpts genuinely do not contain the answer, say: I don't know."
    ),
    "none": (
        "You answer questions about a user's past conversations. "
        "Answer in as few words as possible."
    ),
}

DEFAULT_PROMPT_MODE = "soft"

# Retained so existing callers keep working; equal to the soft variant.
SYSTEM_PROMPT = SYSTEM_PROMPTS[DEFAULT_PROMPT_MODE]

ABSTENTION_STRING = "i don't know"

# Models refuse in whatever words they like, regardless of the phrase they were told to
# use. Measured on real output: "I can't answer that", "I can't help you with that", "I
# don't have any information about...", "I don't have access to...". Matching only the
# sanctioned phrase scored all of those as wrong *answers*, moving grep's answered_rate
# from 0.29 to 0.22 and rag's from 0.29 to 0.25.
#
# **This list will never be exactly right, and that is the point worth recording.** Refusal
# detection by string matching is irreducibly fuzzy. "I don't have personal opinions, but I
# can provide..." is a partial refusal; "I'm happy to help! How..." is a deflection that is
# neither a refusal nor an answer. Any line drawn here is a judgement call, and moving it
# moves `answered_rate` and `accuracy_given_answered`.
#
# Measured sensitivity to where that line is drawn, across three versions of this list:
#
#   answered_rate          0.970 -> 0.740   moved 0.23
#   accuracy_given_answered 0.175 -> 0.216   moved 0.04
#   accuracy_over_all       0.170 -> 0.160   moved 0.01
#
# So `accuracy_over_all` is roughly 23x more stable, NOT immune. It moves because
# `score_response` gives abstention precedence over a keyword match, so widening the
# refusal set can reclassify a response that contained the right answer next to a refusal
# phrase. That precedence rule is still correct -- a hedged non-answer is not an answer --
# but it does couple the honest metric to this judgement call, weakly.
#
# The defensible claim is therefore the quantitative one, not "immune".
ABSTENTION_MARKERS = (
    "i don't know",
    "i do not know",
    "i can't answer",
    "i cannot answer",
    "i can't help",
    "i cannot help",
    "i'm not able to",
    "i am not able to",
    "unable to answer",
    "don't have that information",
    "do not have that information",
    "no information about",
    "not mentioned",
    "isn't mentioned",
    "not specified",
    "don't have any information",
    "do not have any information",
    "don't have access",
    "do not have access",
    "can't provide information",
    "cannot provide information",
)


def is_abstention(text: str) -> bool:
    """Whether a response is a refusal, in any phrasing the reader actually uses."""
    lowered = text.strip().lower()
    return any(marker in lowered for marker in ABSTENTION_MARKERS)

DEFAULT_MAX_NEW_TOKENS = 32


def resolve_device(preference: str = "auto") -> str:
    if preference != "auto":
        return preference
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class Generation:
    text: str
    prompt_tokens: int
    generated_tokens: int

    @property
    def abstained(self) -> bool:
        """Whether the reader declined.

        Substring rather than equality: a small instruct model reliably produces the
        sanctioned phrase but not reliably alone, and scoring "I don't know." as an attempt
        would inflate `answered_rate` with non-answers.
        """
        return is_abstention(self.text)


class Reader:
    def __init__(
        self,
        model: str = READER_MODEL,
        revision: str = READER_REVISION,
        device: str = "auto",
        dtype: str = "auto",
        prompt_mode: str = DEFAULT_PROMPT_MODE,
    ) -> None:
        self.model_name = model
        self.revision = revision
        if prompt_mode not in SYSTEM_PROMPTS:
            raise ValueError(
                f"unknown prompt_mode {prompt_mode!r}; expected one of {list(SYSTEM_PROMPTS)}"
            )
        self.prompt_mode = prompt_mode
        self.device = resolve_device(device)
        self.dtype = dtype
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            torch_dtype = (
                torch.float32
                if self.device == "cpu"
                else (torch.bfloat16 if self.dtype == "auto" else getattr(torch, self.dtype))
            )
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, revision=self.revision)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name, revision=self.revision, dtype=torch_dtype
            ).to(self.device)
            self._model.eval()
        return self._model, self._tokenizer

    @property
    def model(self):
        return self._load()[0]

    @property
    def tokenizer(self):
        return self._load()[1]

    def build_prompt(self, context: str, question: str) -> str:
        """Assemble the chat prompt. An empty context yields a question-only prompt.

        Arm D relies on that branch: with no context there must be no 'here are the
        relevant excerpts' preamble either, or it would be answering a question about
        excerpts it was never given.
        """
        _, tokenizer = self._load()
        if context:
            user = f"Here are excerpts from past conversations:\n\n{context}\n\nQuestion: {question}"
        else:
            user = f"Question: {question}"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPTS[self.prompt_mode]},
            {"role": "user", "content": user},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def generate(
        self,
        context: str,
        question: str,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> Generation:
        import torch

        model, tokenizer = self._load()
        prompt = self.build_prompt(context, question)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        prompt_tokens = int(inputs["input_ids"].shape[-1])

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        generated = out[0][prompt_tokens:]
        return Generation(
            text=tokenizer.decode(generated, skip_special_tokens=True).strip(),
            prompt_tokens=prompt_tokens,
            generated_tokens=int(generated.shape[-1]),
        )

    @property
    def provenance(self) -> dict:
        return {
            "reader_model": self.model_name,
            "reader_revision": self.revision,
            "reader_device": self.device,
            # Recorded because it changed in-context extraction from 38% to 98%. Two rows
            # with different prompt modes are not comparable.
            "prompt_mode": self.prompt_mode,
        }


@lru_cache(maxsize=1)
def shared_reader() -> Reader:
    """Process-wide reader, so a 2.5GB model is loaded once per run."""
    return Reader(device=os.environ.get("WMB_DEVICE", "auto"))
