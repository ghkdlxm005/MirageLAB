"""
generate_etymology.py
---------------------
TEPS 어휘 어원(etymology) 생성기

1차: Wiktionary API (검증된 어원학 DB, 무료, API 키 불필요)
2차: Groq LLM fallback (Wiktionary에 없는 단어)

결과: _data/vocab_etymology.json → 퀴즈·포스트에서 사용

실행: python generate_etymology.py
환경변수: GROQ_API_KEY
"""

import ast, re, json, os, sys, time, urllib.request
from groq import Groq

BATCH_LLM = 10   # LLM fallback 배치 크기
WIKT_DELAY = 0.3  # Wiktionary 요청 간격(초) — 서버 부하 방지

LANG_MAP = {
    'la':  '라틴어',
    'grc': '고대 그리스어',
    'fr':  '프랑스어',
    'fro': '고대 프랑스어',
    'ang': '고대 영어',
    'enm': '중세 영어',
    'de':  '독일어',
    'non': '고대 노르드어',
    'gem': '게르만어',
    'it':  '이탈리아어',
    'es':  '스페인어',
    'ar':  '아랍어',
    'sa':  '산스크리트어',
    'pie': '인도유럽조어',
    'xno': '앵글로-노르만어',
    'ML':  '중세 라틴어',
    'NL':  '신 라틴어',
}


# ── Wiktionary ───────────────────────────────────────────────────────────────

def _wikt_raw(word: str) -> str | None:
    """Wiktionary 위키텍스트 원문 반환."""
    url = (
        "https://en.wiktionary.org/w/api.php"
        f"?action=query&titles={urllib.request.quote(word)}"
        "&prop=revisions&rvprop=content&rvslots=main&format=json&formatversion=2"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "MirageLAB-TEPS-Bot/1.0 (https://github.com/ghkdlxm005/MirageLAB)"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())
        pages = data["query"]["pages"]
        if not pages or pages[0].get("missing"):
            return None
        return pages[0]["revisions"][0]["slots"]["main"]["content"]
    except Exception:
        return None


def _parse_etym(wikitext: str) -> str | None:
    """위키텍스트에서 Etymology 섹션 파싱 → 한국어 어원 문자열."""
    m = re.search(
        r'===?Etymology\s*\d*===?\n(.*?)(?=\n===?|\n==\w|\Z)',
        wikitext, re.DOTALL
    )
    if not m:
        return None
    etym = m.group(1)[:800]

    # inh/der/bor 템플릿: 언어 + 원형단어 + (뜻) 추출
    # 예: {{der|en|la|vīrulentus|t=poisonous}} → 라틴어 vīrulentus(poisonous)
    sources = []
    for tm in re.finditer(
        r'\{\{(?:inh|der|bor)\|en\|(\w+)\|([^|{}\n]+)(?:\|(?:t|gloss)=([^|{}\n}]+))?',
        etym
    ):
        lang = LANG_MAP.get(tm.group(1), tm.group(1))
        term = tm.group(2).strip().lstrip('-').strip()
        meaning = tm.group(3).strip() if tm.group(3) else ''
        if not term or term.startswith('|'):
            continue
        if meaning:
            sources.append(f"{lang} {term}({meaning})")
        else:
            sources.append(f"{lang} {term}")

    if len(sources) >= 2:
        return f"{sources[0]}에서 유래, {sources[1]}에서 유래."
    if len(sources) == 1:
        return f"{sources[0]}에서 유래."

    # 폴백: 마크업 제거 후 원문 텍스트
    text = re.sub(r'\{\{[^}]+\}\}', '', etym)
    text = re.sub(r'\[\[(?:[^\]|]*\|)?([^\]]+)\]\]', r'\1', text)
    text = re.sub(r"'''?", '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = ' '.join(text.split()).strip()
    return text[:180] if len(text) > 15 else None


def get_wiktionary(word: str) -> str | None:
    raw = _wikt_raw(word)
    if not raw:
        return None
    return _parse_etym(raw)


# ── LLM fallback ─────────────────────────────────────────────────────────────

def fetch_llm_batch(client: Groq, words: list) -> dict:
    """words: [(word, meaning, pos), ...] → {word: etymology_str}"""
    word_list = "\n".join(w for w, m, p in words)
    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": (
                    "Output only plain text lines: 'word: etymology'\n"
                    "No JSON, no markdown, no numbering.\n"
                    "Etymology must be in Korean. Include: source language name, "
                    "original root word with meaning in parentheses, affix breakdown if applicable.\n"
                    "Example: sovereignty: 중세 라틴어 superanus(최상위의)에서 유래. super(위) + -anus(형용사 접미사)."
                ),
            },
            {"role": "user", "content": f"Write etymology for each:\n{word_list}"},
        ],
        temperature=0.2,
        max_tokens=800,
    )
    raw = resp.choices[0].message.content.strip()
    result = {}
    for line in raw.splitlines():
        if ': ' in line:
            word, etym = line.split(': ', 1)
            word = word.strip().strip('"').strip('*').strip()
            etym = etym.strip()
            if word and etym:
                result[word] = etym
    return result


# ── 메인 ─────────────────────────────────────────────────────────────────────

def load_vocab():
    with open("teps_generator.py", "r", encoding="utf-8") as f:
        content = f.read()
    m = re.search(r"TEPS_VOCABULARY\s*=\s*(\[.*?\])", content, re.DOTALL)
    return ast.literal_eval(m.group(1))


def main():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY 환경변수가 필요합니다.")
        sys.exit(1)

    client = Groq(api_key=api_key)
    vocab = load_vocab()
    print(f"총 {len(vocab)}개 단어 처리 시작 (Wiktionary 우선, LLM fallback)")

    out_path = "_data/vocab_etymology.json"
    # 기존 파일 초기화 (LLM 생성 데이터 폐기, Wiktionary로 새로 생성)
    result = {}

    llm_queue = []  # Wiktionary 없는 단어 → LLM 배치 처리

    # ── Step 1: Wiktionary ──────────────────────────────────────────────────
    print("\n[1/2] Wiktionary 조회 중...")
    for i, (word, meaning, pos) in enumerate(vocab):
        etym = get_wiktionary(word)
        if etym:
            result[word] = etym
            status = "✓"
        else:
            llm_queue.append((word, meaning, pos))
            status = "–"

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(vocab)} 완료 (Wiktionary {len(result)}개, LLM 대기 {len(llm_queue)}개)")
            # 중간 저장
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

        time.sleep(WIKT_DELAY)

    print(f"\nWiktionary: {len(result)}개 / LLM 필요: {len(llm_queue)}개")

    # ── Step 2: LLM fallback ────────────────────────────────────────────────
    if llm_queue:
        print(f"\n[2/2] LLM fallback 처리 중 ({len(llm_queue)}개)...")
        total_batches = (len(llm_queue) + BATCH_LLM - 1) // BATCH_LLM
        for i in range(0, len(llm_queue), BATCH_LLM):
            batch = llm_queue[i:i + BATCH_LLM]
            batch_num = i // BATCH_LLM + 1
            preview = [w for w, _, _ in batch[:3]]
            print(f"  배치 {batch_num}/{total_batches}: {preview}... ", end="", flush=True)
            try:
                etyms = fetch_llm_batch(client, batch)
                result.update(etyms)
                print(f"{len(etyms)}개 완료")
            except Exception as e:
                print(f"실패: {e}")

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            time.sleep(0.5)

    # ── 최종 저장 ────────────────────────────────────────────────────────────
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    wikt_count = sum(1 for w, _, _ in vocab if result.get(w) and (w, _, _) not in llm_queue)
    print(f"\n완료! 총 {len(result)}개 저장 → {out_path}")
    print(f"  Wiktionary: {len(result) - len(llm_queue) + (len(llm_queue) - sum(1 for w,_,_ in llm_queue if w not in result))}개")
    print(f"  LLM fallback: {sum(1 for w,_,_ in llm_queue if w in result)}개")


if __name__ == "__main__":
    main()
