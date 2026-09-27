# 외부 구성요소 및 라이선스

이 저장소의 자체 코드는 [Apache-2.0](LICENSE)으로 제공한다. 외부 엔진, 모델, 폰트, API 서비스에는 각각의 라이선스·이용 조건이 적용된다. 원고와 번역 결과에 대한 권리는 이 소프트웨어 라이선스로 부여되지 않는다.

## 외부 엔진

| 구성요소 | 사용 버전 | 프로젝트 및 라이선스 | 배포 방식 |
|---|---|---|---|
| Koharu | 0.83.5 | [koharu-rs/koharu](https://github.com/koharu-rs/koharu), MIT OR Apache-2.0 | 설치 시 원본 소스 다운로드·로컬 빌드 |
| MangaTranslator | v1.24.7 | [meangrinch/MangaTranslator](https://github.com/meangrinch/MangaTranslator), Apache-2.0 | 설치 시 원본 소스 다운로드·별도 가상환경 구성 |
| Noto Sans KR | 설치 시 다운로드 | [Google Fonts](https://github.com/google/fonts/tree/main/ofl/notosanskr), SIL Open Font License 1.1 | 폰트와 OFL.txt를 함께 다운로드 |

`engines/koharu/quality.patch`는 Koharu의 본문 검출·마스크 범위·한국어 식자를 수정한 패치이며, 원본 코드 문맥을 포함한다. 해당 원본 부분에 대한 MIT 저작권·허가 고지를 아래에 보존한다. 설치 과정에서의 CUDA 드라이버 하한 변경과 `ct_batch` 배치 실행기 통합은 이 프로젝트의 설치 스크립트에서 별도로 적용된다. 외부 소스의 라이선스 파일을 제거하지 않는다.

### Koharu MIT 고지

```text
MIT License

Copyright (c) 2025-2026 Mayo Takanashi and Koharu contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 모델·런타임·온라인 서비스

- RF-DETR, Comic Text Detector, PaddleOCR-VL, manga-ocr, LaMa, FLUX.2 Klein 등은 선택한 엔진과 설정에 따라 다운로드된다. 모델별 모델 카드의 라이선스·접근 승인·사용 제한을 확인해야 한다. 이 저장소는 모델 가중치를 재배포하지 않는다.
- CUDA, PyTorch 및 그 종속성은 각각의 배포 조건을 따른다. 설치 스크립트가 의존성을 다운로드한다는 사실이 재배포 권한이나 특정 GPU의 호환성을 보장하지는 않는다.
- LLM 제공자의 API·OAuth 사용은 해당 서비스의 약관과 계정 권한을 따른다. 서비스 사용료는 별도다.
- Python 앱 의존성은 `pyproject.toml`과 `uv.lock`, 외부 엔진 의존성은 각 엔진의 소스·잠금 파일·라이선스를 확인한다.
