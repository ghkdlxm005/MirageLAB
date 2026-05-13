"""
teps_generator.py
------------------
Groq API (llama-3.3-70b-versatile)를 사용해 매일 TEPS 포스트를 생성하고
_posts/ 폴더에 Jekyll 마크다운 파일로 저장합니다.

실행: python teps_generator.py
환경변수: GROQ_API_KEY
"""

import os
import sys
import re
from datetime import datetime, timezone, timedelta
from groq import Groq

# ── 설정 ─────────────────────────────────────────────────────────────────────
KST = timezone(timedelta(hours=9))
MODEL = "llama-3.1-8b-instant"
POSTS_DIR = "_posts"

SYSTEM_PROMPT = """당신은 TEPS(Test of English Proficiency developed by Seoul National University) 전문 강사입니다.
매일 학습자를 위한 TEPS 학습 콘텐츠를 제공합니다.

반드시 지켜야 할 규칙:
1. 출력에 사용 가능한 문자는 영어 알파벳과 한국어 한글, 숫자, 일반 문장부호만 허용합니다.
2. 한자, 힌디 문자, 아랍 문자, 일본어, IPA 발음기호 등 그 외 모든 문자는 절대 사용하지 마세요.
3. 이모티콘(emoji)은 절대 사용하지 마세요.
4. 독해 지문은 반드시 영어로 작성하세요. 한국어 지문은 절대 안 됩니다.
5. 출력은 반드시 지정된 마크다운 형식을 엄격히 따르세요."""

USER_PROMPT_TEMPLATE = """오늘 날짜: {date}

다음 형식으로 TEPS 일일 학습 콘텐츠를 작성해주세요.
회사 쉬는 시간 10~15분 안에 볼 수 있는 분량으로 만들어주세요.

주의: 영어와 한국어만 사용. 이모티콘 금지. 독해 지문은 반드시 영어로 작성.

---

## 오늘의 TEPS 단어 (5개)

각 단어마다 아래 형식으로 정확히 작성하세요:

**1. [단어]** /[IPA 발음기호]/ ([한글 발음 예: 비질런트]) _[품사]_
- 뜻: [한국어 의미]
- 예문: [영어 예문]
- 해석: [예문 한국어 해석]

**2. [단어]** /[IPA 발음기호]/ ([한글 발음]) _[품사]_
- 뜻: [한국어 의미]
- 예문: [영어 예문]
- 해석: [예문 한국어 해석]

**3. [단어]** /[IPA 발음기호]/ ([한글 발음]) _[품사]_
- 뜻: [한국어 의미]
- 예문: [영어 예문]
- 해석: [예문 한국어 해석]

**4. [단어]** /[IPA 발음기호]/ ([한글 발음]) _[품사]_
- 뜻: [한국어 의미]
- 예문: [영어 예문]
- 해석: [예문 한국어 해석]

**5. [단어]** /[IPA 발음기호]/ ([한글 발음]) _[품사]_
- 뜻: [한국어 의미]
- 예문: [영어 예문]
- 해석: [예문 한국어 해석]

---

## 단어 확인 퀴즈

위 단어 5개를 활용한 빈칸 채우기 문제 3개를 영어로 만드세요.

**Q1.** [빈칸이 포함된 영어 문장]
(A) [선택지1]  (B) [선택지2]  (C) [선택지3]  (D) [선택지4]

**Q2.** [빈칸이 포함된 영어 문장]
(A) [선택지1]  (B) [선택지2]  (C) [선택지3]  (D) [선택지4]

**Q3.** [빈칸이 포함된 영어 문장]
(A) [선택지1]  (B) [선택지2]  (C) [선택지3]  (D) [선택지4]

<details>
<summary>퀴즈 정답 보기</summary>

Q1 정답: [(정답)] — [한국어 해설]

Q2 정답: [(정답)] — [한국어 해설]

Q3 정답: [(정답)] — [한국어 해설]

</details>

---

## 오늘의 독해

지문은 반드시 영어로 작성하세요. 시사, 경제, 과학, 환경 주제 중 선택. TEPS 독해 수준.

**지문 (영어로 작성):**
[100~150단어 분량의 영어 지문]

**문제 1.** [영어 질문]
(A) [영어 선택지1]
(B) [영어 선택지2]
(C) [영어 선택지3]
(D) [영어 선택지4]

**문제 2.** [영어 질문]
(A) [영어 선택지1]
(B) [영어 선택지2]
(C) [영어 선택지3]
(D) [영어 선택지4]

<details>
<summary>정답 및 해설 보기</summary>

**문제 1 정답:** [(정답)] — [한국어 해설]

**문제 2 정답:** [(정답)] — [한국어 해설]

**지문 해석:**
[전체 지문 한국어 해석]

</details>

---

## 오늘의 문법 포인트

**주제:** [문법 주제]

[2~3문장 한국어 설명]

**올바른 예문:**
- [영어 예문] — [한국어 해석]

**틀린 예문과 수정:**
- 틀림: [틀린 영어 예문]
- 수정: [올바른 영어 예문] — [왜 틀렸는지 한국어로 설명]

---"""


def get_kst_date() -> datetime:
    return datetime.now(KST)


def generate_content(client: Groq, date_str: str) -> str:
    prompt = USER_PROMPT_TEMPLATE.format(date=date_str)

    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.8,
        max_tokens=2048,
    )

    return completion.choices[0].message.content


def build_front_matter(date: datetime) -> str:
    date_str = date.strftime("%Y-%m-%d")
    time_str = date.strftime("%Y-%m-%d %H:%M:%S +0900")

    return f"""---
layout: post
title: "TEPS 일일 학습 — {date_str}"
date: {time_str}
categories: teps
tags: [TEPS, 영어, 단어, 독해, 문법]
---
"""


def save_post(content: str, date: datetime) -> str:
    os.makedirs(POSTS_DIR, exist_ok=True)

    date_str = date.strftime("%Y-%m-%d")
    filename = f"{date_str}-teps-daily.md"
    filepath = os.path.join(POSTS_DIR, filename)

    front_matter = build_front_matter(date)
    full_content = front_matter + "\n" + content

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(full_content)

    return filepath


def main():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY 환경변수가 설정되지 않았습니다.")
        sys.exit(1)

    client = Groq(api_key=api_key)
    today = get_kst_date()
    date_str = today.strftime("%Y년 %m월 %d일")

    print(f"[{date_str}] TEPS 포스트 생성 시작...")

    try:
        content = generate_content(client, date_str)
        filepath = save_post(content, today)
        print(f"포스트 저장 완료: {filepath}")
    except Exception as e:
        print(f"포스트 생성 실패: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
