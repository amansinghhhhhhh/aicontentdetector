"""Quick smoke test for the ML core: score sample human / AI / mixed texts."""

import sys

from ml.detect import analyze_text

HUMAN_TEXT = """\
okay so i finally tried that new ramen place downtown and honestly? not worth the hype.
the broth was kinda oily and my noodles were overcooked lol. i asked for extra chashu and
they forgot it entirely. my friend got the veggie one and she liked hers better than i
did. would i go back? probably not, but maybe if they fix the noodles. prices are fine
though, like 14 bucks for a big bowl. whatever, i'll stick to my usual spot on 5th.
"""

AI_TEXT = """\
In today's fast-paced digital landscape, artificial intelligence has emerged as a
transformative force across numerous industries. Organizations are increasingly leveraging
AI-driven solutions to streamline operations, enhance decision-making processes, and unlock
valuable insights from vast datasets. Moreover, by embracing these cutting-edge technologies,
businesses can maintain a competitive edge while delivering superior value to their customers.
It is important to note that successful implementation requires careful planning, robust
infrastructure, and a commitment to continuous learning. Ultimately, the future of innovation
lies in the thoughtful integration of intelligent systems into everyday workflows.
"""

MIXED_TEXT = """\
In the modern workplace, effective communication is essential for achieving organizational
goals and fostering collaboration among team members. Now let me tell you what actually
happened at my office last week — our manager sent this long email about "synergy" and
nobody understood a single paragraph, so we just replied "sounds good" like we always do.
Great leaders understand the importance of active listening and empathetic engagement.
Anyway, my coworker Priya actually called the client back and fixed the whole thing in
ten minutes. Pretty sure she deserves a raise, not another PowerPoint template.
"""


def show(title: str, text: str) -> None:
    result = analyze_text(text)
    print(f"\n=== {title} ===")
    print(
        f"ai_percent={result['ai_percent']}  verdict={result['verdict']}  "
        f"confidence={result['confidence']}  time={result['elapsed_ms']}ms"
    )
    print(f"class_percent={result['class_percent']}")
    print(f"stats={result['stats']}")
    for w in result["warnings"]:
        print(f"warning: {w}")
    for s in result["sentences"]:
        snippet = s["text"][:70].replace("\n", " ")
        print(f"  [{s['score']:5.1f}] {snippet}{'...' if len(s['text']) > 70 else ''}")


def main() -> int:
    print("Loading model (first run downloads ~1.7 GB)...")
    show("HUMAN (expect low %)", HUMAN_TEXT)
    show("AI (expect high %)", AI_TEXT)
    show("MIXED (expect middle %)", MIXED_TEXT)

    h = analyze_text(HUMAN_TEXT)["ai_percent"]
    a = analyze_text(AI_TEXT)["ai_percent"]
    ok = h < 35 and a > 60 and a - h > 40
    print(f"\nSMOKE RESULT: {'PASS' if ok else 'FAIL'} (human={h}, ai={a})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
