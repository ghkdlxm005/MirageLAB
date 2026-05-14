"""
generate_etymology.py
---------------------
TEPS 어휘 1287개에 대한 어원(etymology)을 Groq API로 일괄 생성해
_data/vocab_etymology.json 으로 저장합니다.

실행: python generate_etymology.py
환경변수: GROQ_API_KEY
"""

import ast, re, json, os, sys, time
from groq import Groq

BATCH = 25  # 한 번에 처리할 단어 수

def load_vocab():
    with open("teps_generator.py", "r", encoding="utf-8") as f:
        content = f.read()
    match = re.search(r"TEPS_VOCABULARY\s*=\s*(\[.*?\])", content, re.DOTALL)
    return ast.literal_eval(match.group(1))

def fetch_etymologies(client: Groq, words: list) -> dict:
    """words: [(word, meaning, pos), ...] → {word: etymology_str}"""
    word_list = "\n".join(f"{i+1}. {w}" for i, (w, m, p) in enumerate(words))
    prompt = f"""For each English word below, write a single short Korean sentence (max 40 chars) explaining its Latin/Greek/Old English origin.
Return ONLY valid JSON in this format: {{"word": "어원 설명"}}
No extra text, no markdown, just the JSON object.

Words:
{word_list}"""

    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You output only valid JSON. No markdown, no explanation."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=1200,
    )
    raw = resp.choices[0].message.content.strip()
    # JSON만 추출
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {}
    return json.loads(match.group())

def main():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY 환경변수가 필요합니다.")
        sys.exit(1)

    client = Groq(api_key=api_key)
    vocab = load_vocab()
    print(f"총 {len(vocab)}개 단어 처리 시작...")

    # 이미 생성된 데이터 로드 (재시작 시 이어서 진행)
    out_path = "_data/vocab_etymology.json"
    result = {}
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            result = json.load(f)
        print(f"  기존 {len(result)}개 로드, 이어서 진행합니다.")

    todo = [(w, m, p) for w, m, p in vocab if w not in result]
    total_batches = (len(todo) + BATCH - 1) // BATCH

    for i in range(0, len(todo), BATCH):
        batch = todo[i:i + BATCH]
        batch_num = i // BATCH + 1
        print(f"  배치 {batch_num}/{total_batches}: {[w for w,_,_ in batch[:3]]}... ", end="", flush=True)
        try:
            etyms = fetch_etymologies(client, batch)
            result.update(etyms)
            print(f"{len(etyms)}개 완료")
        except Exception as e:
            print(f"실패: {e}")

        # 중간 저장 (배치마다)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        time.sleep(0.5)  # API rate limit 방지

    print(f"\n완료! 총 {len(result)}개 어원 저장 → {out_path}")

if __name__ == "__main__":
    main()
