"""
generate_etymology.py
---------------------
TEPS 어휘 IPA 발음 + 어원(etymology) 검증 생성기

1차: Wiktionary API (검증된 어원학 DB, 무료, API 키 불필요)
     - IPA 발음기호 추출
     - 어원(etymology) 추출
2차: Groq LLM fallback (Wiktionary에 없는 단어)

결과: _data/vocab_verified.json → {word: {ipa, etymology}}
     _data/vocab_etymology.json → {word: etymology} (하위 호환)

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
    'nl':  '네덜란드어',
    'pt':  '포르투갈어',
    'ru':  '러시아어',
    'ja':  '일본어',
    'zh':  '중국어',
    'he':  '히브리어',
    'af':  '아프리칸스어',
    'sv':  '스웨덴어',
    'da':  '덴마크어',
    'no':  '노르웨이어',
}


# ── Wiktionary ───────────────────────────────────────────────────────────────

def _wikt_raw(word: str) -> str | None:
    """Wiktionary 위키텍스트 원문 반환."""
    url = (
        "https://en.wiktionary.org/w/api.php"
        f"?action=query&titles={urllib.request.quote(word)}"
        "&prop=revisions&rvprop=content&rvslots=main&format=json&formatversion=2"
    )
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "MirageLAB-TEPS-Bot/1.0 (https://github.com/ghkdlxm005/MirageLAB)"}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())
        pages = data["query"]["pages"]
        if not pages or pages[0].get("missing"):
            return None
        return pages[0]["revisions"][0]["slots"]["main"]["content"]
    except Exception:
        return None


def _parse_ipa(wikitext: str) -> str | None:
    """위키텍스트에서 IPA 발음기호 추출.

    예: {{IPA|en|/ˈpʌb.lɪk/}} → /ˈpʌb.lɪk/
    미국식(GenAm) 우선, 없으면 영국식(RP)
    """
    # {{a|GA}} ... {{IPA|en|/ipa/}} 패턴 — GenAm 우선
    # 먼저 General American 섹션에서 시도
    ga_section = re.search(r'\{\{a\|G[Aa][^}]*\}\}.*?(?=\n\n|\n==|\{\{a\|)', wikitext, re.DOTALL)
    if ga_section:
        m = re.search(r'\{\{IPA\|en\|[^}]*?(/[^/|}{]+/)[^}]*?\}\}', ga_section.group(0))
        if m:
            return m.group(1)

    # 모든 IPA 템플릿에서 첫 번째
    m = re.search(r'\{\{IPA\|en\|[^}]*?(/[^/|}{]+/)[^}]*?\}\}', wikitext)
    if m:
        return m.group(1)

    # {{IPA|/ipa/}} 형식 (lang 없는 경우)
    m = re.search(r'\{\{IPA\|(/[^/|}{]+/)', wikitext)
    if m:
        return m.group(1)

    # {{IPAc-en|...}} 형식 파싱
    m = re.search(r'\{\{IPAc-en\|([^}]+)\}\}', wikitext)
    if m:
        parts = m.group(1).split('|')
        # 발음기호 조각들을 합침 (숫자·따옴표 등 필터링)
        symbols = [p.strip() for p in parts if p.strip() and not p.strip().startswith("'")]
        if symbols:
            return '/' + ''.join(s for s in symbols if s not in ('audio', 'lang')) + '/'

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
        return f"{sources[0]}에서 유래, {sources[1]}을 거쳐 발전."
    if len(sources) == 1:
        return f"{sources[0]}에서 유래."

    # 폴백: 마크업 제거 후 원문 텍스트
    text = re.sub(r'\{\{[^}]+\}\}', '', etym)
    text = re.sub(r'\[\[(?:[^\]|]*\|)?([^\]]+)\]\]', r'\1', text)
    text = re.sub(r"'''?", '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = ' '.join(text.split()).strip()
    return text[:180] if len(text) > 15 else None


def get_wiktionary(word: str) -> dict:
    """Wiktionary에서 IPA + 어원 동시 추출. 반환: {ipa, etymology} (없으면 None 값)"""
    raw = _wikt_raw(word)
    if not raw:
        return {"ipa": None, "etymology": None}
    return {
        "ipa": _parse_ipa(raw),
        "etymology": _parse_etym(raw),
    }


# ── LLM fallback ─────────────────────────────────────────────────────────────

def fetch_llm_batch(client: Groq, words: list) -> dict:
    """words: [(word, meaning, pos), ...] → {word: {ipa, etymology}}"""
    word_list = "\n".join(f"{w} ({p}) — {m}" for w, m, p in words)
    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": (
                    "Output only plain text lines in this exact format:\n"
                    "word|/IPA/|etymology\n\n"
                    "Rules:\n"
                    "- IPA: standard American English IPA notation (e.g. /ˈpʌb.lɪk/)\n"
                    "- etymology: Korean text. Include source language, original root with meaning in parentheses.\n"
                    "- Example: public|/ˈpʌb.lɪk/|라틴어 publicus(공공의)에서 유래. pub-(성인) + -licus(형용사 접미사).\n"
                    "- No JSON, no markdown, no extra lines.\n"
                    "- No Chinese characters. Korean only."
                ),
            },
            {"role": "user", "content": f"Provide IPA and Korean etymology for each word:\n{word_list}"},
        ],
        temperature=0.2,
        max_tokens=1200,
    )
    raw = resp.choices[0].message.content.strip()
    result = {}
    for line in raw.splitlines():
        parts = line.split('|', 2)
        if len(parts) == 3:
            word = parts[0].strip().strip('"').strip('*').strip()
            ipa = parts[1].strip()
            etym = parts[2].strip()
            if word and ipa:
                result[word] = {
                    "ipa": ipa if ipa.startswith('/') else f'/{ipa}/',
                    "etymology": etym if etym else None,
                }
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
    print(f"총 {len(vocab)}개 단어 처리 시작 (Wiktionary IPA+어원 우선, LLM fallback)")

    verified_path = "_data/vocab_verified.json"
    etym_path = "_data/vocab_etymology.json"   # 하위 호환용

    os.makedirs("_data", exist_ok=True)

    verified = {}   # {word: {ipa, etymology}}
    llm_queue = []  # IPA 또는 어원이 없는 단어 → LLM 보완

    # ── Step 1: Wiktionary ──────────────────────────────────────────────────
    print("\n[1/2] Wiktionary 조회 중 (IPA + 어원)...")
    ipa_count = 0
    etym_count = 0

    for i, (word, meaning, pos) in enumerate(vocab):
        data = get_wiktionary(word)
        verified[word] = data

        if data["ipa"]:
            ipa_count += 1
        if data["etymology"]:
            etym_count += 1

        # IPA도 없고 어원도 없으면 LLM 큐
        if not data["ipa"] and not data["etymology"]:
            llm_queue.append((word, meaning, pos))

        status = f"IPA={'✓' if data['ipa'] else '–'} 어원={'✓' if data['etymology'] else '–'}"
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(vocab)} 완료 | IPA {ipa_count}개 | 어원 {etym_count}개 | LLM 대기 {len(llm_queue)}개")
            # 중간 저장
            with open(verified_path, "w", encoding="utf-8") as f:
                json.dump(verified, f, ensure_ascii=False, indent=2)

        time.sleep(WIKT_DELAY)

    print(f"\nWiktionary 완료 → IPA: {ipa_count}개 / 어원: {etym_count}개 / LLM 필요: {len(llm_queue)}개")

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
                batch_result = fetch_llm_batch(client, batch)
                for word, data in batch_result.items():
                    # LLM이 준 것으로 보완 (Wiktionary에서 일부만 있을 수도 있음)
                    if word in verified:
                        if not verified[word]["ipa"] and data.get("ipa"):
                            verified[word]["ipa"] = data["ipa"]
                        if not verified[word]["etymology"] and data.get("etymology"):
                            verified[word]["etymology"] = data["etymology"]
                    else:
                        verified[word] = data
                print(f"{len(batch_result)}개 완료")
            except Exception as e:
                print(f"실패: {e}")

            with open(verified_path, "w", encoding="utf-8") as f:
                json.dump(verified, f, ensure_ascii=False, indent=2)
            time.sleep(0.5)

    # ── 최종 저장 ────────────────────────────────────────────────────────────
    with open(verified_path, "w", encoding="utf-8") as f:
        json.dump(verified, f, ensure_ascii=False, indent=2)

    # 하위 호환: vocab_etymology.json도 업데이트
    etym_only = {w: d["etymology"] for w, d in verified.items() if d.get("etymology")}
    with open(etym_path, "w", encoding="utf-8") as f:
        json.dump(etym_only, f, ensure_ascii=False, indent=2)

    final_ipa = sum(1 for d in verified.values() if d.get("ipa"))
    final_etym = sum(1 for d in verified.values() if d.get("etymology"))
    print(f"\n완료! 총 {len(verified)}개 저장 → {verified_path}")
    print(f"  IPA 확보: {final_ipa}개 ({final_ipa/len(vocab)*100:.1f}%)")
    print(f"  어원 확보: {final_etym}개 ({final_etym/len(vocab)*100:.1f}%)")
    print(f"  하위 호환 파일: {etym_path}")


if __name__ == "__main__":
    main()
