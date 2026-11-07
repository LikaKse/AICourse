import os
import sys

from google import genai
from google.genai import errors

# Модель Gemini, которой отправляем запрос (актуальная Flash-модель)
MODEL = "gemini-3.8-flash"


def main():
    # Читаем API-ключ из переменной окружения GEMINI_API_KEY
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Ошибка: не задан API-ключ. Установите переменную окружения GEMINI_API_KEY.")
        sys.exit(1)

    # Получаем текстовый запрос от пользователя
    prompt = input("Введите ваш запрос: ").strip()
    if not prompt:
        print("Ошибка: запрос не может быть пустым.")
        sys.exit(1)

    # Создаём клиента для работы с Gemini API
    client = genai.Client(api_key=api_key)

    try:
        # Отправляем запрос пользователя в модель Gemini
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
        )
    except errors.APIError as e:
        # Ошибка, которую вернул сам API: неверный ключ, превышен лимит, неизвестная модель и т.п.
        print(f"Ошибка API Gemini (код {e.code}): {e.message}")
        sys.exit(1)
    except Exception as e:
        # Любая другая ошибка, например нет подключения к интернету
        print(f"Не удалось выполнить запрос: {e}")
        sys.exit(1)

    # Выводим текст ответа модели в консоль
    print("\nОтвет модели:")
    print(response.text)


if __name__ == "__main__":
    main()
