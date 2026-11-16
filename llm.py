"""Language model providers. Both options are free.

none   : no model. Answers are extractive (the best-matching policy
         sentences, quoted with citations). Runs anywhere.
ollama : a local open model through Ollama (https://ollama.com), e.g.
         `ollama pull llama3.2`. No API key, nothing leaves your computer.
"""
import json
import urllib.request


class OllamaLLM:
    def __init__(self, model: str = "llama3.2", host: str = "http://localhost:11434"):
        self.model, self.host = model, host

    def generate(self, prompt: str) -> str:
        body = json.dumps({"model": self.model, "prompt": prompt, "stream": False,
                           "options": {"temperature": 0}}).encode()
        req = urllib.request.Request(f"{self.host}/api/generate", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read())["response"].strip()
        except OSError as e:
            raise RuntimeError(
                f"Could not reach Ollama at {self.host}. Install it from ollama.com, run "
                f"`ollama pull {self.model}`, and make sure the Ollama app is running."
            ) from e


def get_llm(name: str, model: str = "llama3.2"):
    if name == "none":
        return None
    if name == "ollama":
        return OllamaLLM(model)
    raise ValueError("llm must be 'none' or 'ollama'")
