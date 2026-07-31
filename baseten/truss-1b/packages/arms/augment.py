"""Knowledge augmentation: the ingredient whose absence explains the whole negative result.

Allen-Zhu & Li (*Physics of Language Models 3.1*, arXiv 2309.14316) showed that a fact seen
in only one phrasing is **memorized but not extractable** — 0% accuracy on questions about
it, "regardless of subsequent instruction fine-tuning". The model's loss on the training
string drops to near zero and it still cannot answer a question about it.

That is exactly what this repo measured and misread: fact NLL collapsed while recall stayed
at zero, and the conclusion drawn was that the mechanism does not work. The mechanism works.
It was being fed one sentence per fact.

So: before any weight update, expand each fact into many surface forms. Declarative,
question-and-answer, reversed subject and object, third person, embedded in longer prose.
The model generates them itself (SEAL-style, arXiv 2506.10943), which is both the published
approach and the path to putting an RL policy over *what to generate*.

**Generation is sampled, not greedy.** The reader is greedy everywhere else on purpose, to
keep accuracy comparisons deterministic. Greedy decoding here would return N copies of one
sentence, which would look exactly like augmentation and supply none of it.

**Every variant is validated.** A variant that does not contain the answer teaches the model
a different fact, and one identical to its siblings adds nothing. Both are dropped, and the
drop count is reported rather than hidden — a generator quietly failing validation would
present as "augmentation did not help".
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Several instructions rather than one, because a single prompt sampled N times produces
# variations on one syntactic frame. Diversity of *form* is the active ingredient; diversity
# of wording within one form is not.
REWRITE_INSTRUCTIONS = (
    "Rewrite this sentence in different words, keeping every detail exactly the same:",
    "Restate this fact as a short question and its answer:",
    "State this fact plainly in the third person, as if describing someone else:",
    "Mention this fact in the middle of a longer, natural-sounding sentence:",
    "Rephrase this, putting the most important detail at the start of the sentence:",
)

# Deterministic forms, always included alongside the generated ones. A 1B model is an
# unreliable paraphraser, and if every generated variant fails validation these guarantee the
# experiment still tests augmentation rather than testing the generator.
TEMPLATES = (
    "{statement}",
    "Q: {question}\nA: {answer}",
    "{question} {answer}.",
    "The answer to '{question}' is {answer}.",
    "Remember: {question} {answer}.",
)


@dataclass
class AugmentationResult:
    fact_key: str
    variants: list[str]
    generated_kept: int = 0
    generated_dropped: int = 0
    drop_reasons: dict[str, int] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.variants)


def _sample(reader, prompt: str, max_new_tokens: int = 60, temperature: float = 0.9) -> str:
    """One sampled continuation. Kept local so the shared greedy reader stays greedy."""
    import torch

    model, tokenizer = reader._load()
    messages = [
        {"role": "system", "content": "You rewrite facts. Reply with the rewrite only."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(reader.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    generated = out[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def templated(fact) -> list[str]:
    return [
        t.format(statement=fact.statement, question=fact.question, answer=fact.answer)
        for t in TEMPLATES
    ]


def augment(fact, reader=None, n_generated: int = 10, temperature: float = 0.9) -> AugmentationResult:
    """Expand one fact into many surface forms.

    Templates are always present; generated variants are added on top and validated. The
    result records how many generations were dropped and why, so a failing generator is
    visible in the output rather than showing up later as a null result.
    """
    result = AugmentationResult(fact_key=fact.key, variants=templated(fact))
    if reader is None or n_generated <= 0:
        return result

    seen = {v.lower().strip() for v in result.variants}
    for i in range(n_generated):
        instruction = REWRITE_INSTRUCTIONS[i % len(REWRITE_INSTRUCTIONS)]
        try:
            candidate = _sample(
                reader, f"{instruction}\n\n{fact.statement}", temperature=temperature
            )
        except Exception:
            result.generated_dropped += 1
            result.drop_reasons["generation_error"] = (
                result.drop_reasons.get("generation_error", 0) + 1
            )
            continue

        candidate = candidate.split("\n\n")[0].strip()

        if not fact.matches(candidate):
            # Does not contain the answer: it teaches a different fact, or none.
            result.generated_dropped += 1
            result.drop_reasons["missing_answer"] = (
                result.drop_reasons.get("missing_answer", 0) + 1
            )
            continue
        key = candidate.lower().strip()
        if key in seen:
            result.generated_dropped += 1
            result.drop_reasons["duplicate"] = result.drop_reasons.get("duplicate", 0) + 1
            continue
        if len(candidate) < 12:
            result.generated_dropped += 1
            result.drop_reasons["too_short"] = result.drop_reasons.get("too_short", 0) + 1
            continue

        seen.add(key)
        result.variants.append(candidate)
        result.generated_kept += 1

    return result


def augment_all(facts, reader=None, n_generated: int = 10) -> list[AugmentationResult]:
    return [augment(f, reader=reader, n_generated=n_generated) for f in facts]


def training_corpus(results) -> list[str]:
    """Flatten to the text the update actually trains on.

    Interleaved by variant index rather than grouped by fact, so the optimizer never sees a
    long run of one fact's phrasings. Grouping would make each fact a local distribution
    shift, which is the condition that produces the interference this whole exercise is
    trying to avoid.
    """
    out: list[str] = []
    depth = max((len(r.variants) for r in results), default=0)
    for i in range(depth):
        for result in results:
            if i < len(result.variants):
                out.append(result.variants[i])
    return out
