"""День 1. Первый запрос к LLM через API.

Скрипт отправляет один текстовый запрос в Gemini API и выводит ответ модели в консоль.
API-ключ берётся из переменной окружения GEMINI_API_KEY (её можно задать в файле .env).

Запуск:
    python main.py                       # запрос по умолчанию
    python main.py "Ваш вопрос модели"   # свой запрос
"""

import os
import sys
from pathlib import Path

API_KEY_ENV = "GEMINI_API_KEY"
ENV_FILE = Path(__file__).parent / ".env"

# Та же недорогая модель, что и в Днях 3–5 (есть бесплатный уровень)
MODEL = "gemini-3.5-flash-lite"
DEFAULT_PROMPT = "Объясни в двух предложениях, что такое большая языковая модель (LLM)."
REQUEST_TIMEOUT_SEC = 60


def load_env_file(path):
    """Загружает пары KEY=VALUE из .env в os.environ.

    Минимальная замена python-dotenv (в окружении проекта её нет).
    Уже заданные переменные окружения не перезаписываются.
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def main():
    load_env_file(ENV_FILE)

    # 1. Ключ: только из окружения / .env, никогда не хардкодим
    api_key = os.environ.get(API_KEY_ENV, "").strip()
    if not api_key:
        print(f"Ошибка: не задан API-ключ {API_KEY_ENV}.\n"
              f"Скопируйте .env.example в .env и впишите ключ из https://aistudio.google.com/apikey",
              file=sys.stderr)
        sys.exit(1)

    # 2. Текст запроса: из аргументов командной строки или запрос по умолчанию
    prompt = " ".join(sys.argv[1:]).strip() or DEFAULT_PROMPT

    from google import genai
    from google.genai import errors, types

    client = genai.Client(api_key=api_key,
                          http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_SEC * 1000))

    print(f"Модель: {MODEL}")
    print(f"Запрос: {prompt}\n", flush=True)  # flush: заголовок выводится до сообщения об ошибке

    # 3. Отправляем запрос и обрабатываем ошибки
    try:
        response = client.models.generate_content(model=MODEL, contents=prompt)
    except errors.APIError as e:
        # Ошибку вернул сам API: неверный ключ, превышен лимит, неизвестная модель и т.п.
        print(f"Ошибка Gemini API (код {e.code}, {e.status}): {e.message}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        # Сетевые ошибки, таймаут и прочее
        print(f"Не удалось выполнить запрос: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(3)

    # 4. Выводим ответ
    text = response.text
    if not text:
        print("Модель вернула пустой ответ.", file=sys.stderr)
        sys.exit(4)
    print("Ответ модели:")
    print(text)


if __name__ == "__main__":
    main()
