"""Korean localization guidance appended to every translation request."""

KOREAN_GUIDE = """[한국어 만화 번역 지침]
- 일본어 만화 대사를 자연스러운 한국어 구어체 만화 대사로 옮긴다. 직역투를 피한다.
- 말풍선에 들어가도록 간결하게 쓴다. 원문에 없는 설명을 덧붙이지 않는다.
- 인물 관계에 맞게 존댓말/반말을 일관되게 유지한다 (선배·손윗사람에게는 존댓말).
- 호칭은 한국 독자에게 자연스럽게 옮긴다: 先輩→선배, ～さん→이름 또는 ～씨, ～ちゃん/～くん은 문맥에 맞게 생략하거나 애칭으로.
- 말줄임(……), 더듬기(べ、別に→뭐, 별로), 감탄·의성어는 한국어 만화 관습에 맞춘다.
- 고유명사는 일관된 한국어 표기를 쓰고, 아래 작품 용어집이 있으면 반드시 따른다."""


def build_instructions(user_notes: str) -> str:
    notes = user_notes.strip()
    return f"{KOREAN_GUIDE}\n\n[작품 용어집·인물 메모]\n{notes}" if notes else KOREAN_GUIDE
