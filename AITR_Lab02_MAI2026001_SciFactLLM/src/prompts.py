"""Prompt construction from configs/prompts.yaml. Gold labels and ids never enter the prompt."""
import yaml


def load_prompts(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _user(prompts, claim, title, abstract):
    return prompts["user_format"].format(claim=claim, title=title, abstract=abstract)


def build_messages(prompts, template, row, fewshot=None, abstract_override=None):
    system = prompts["system"].strip() + "\n\n" + prompts["templates"][template]["instruction"].strip()
    msgs = [{"role": "system", "content": system}]
    if fewshot is not None:
        for _, ex in fewshot.iterrows():
            msgs.append({"role": "user", "content": _user(prompts, ex["claim"], ex["title"], ex["abstract"])})
            msgs.append({"role": "assistant", "content": ex["gold"]})
    abstract = row["abstract"] if abstract_override is None else abstract_override
    msgs.append({"role": "user", "content": _user(prompts, row["claim"], row["title"], abstract)})
    return msgs
