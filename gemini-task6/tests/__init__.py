"""Офлайн-тесты. Запуск из папки gemini-task6: ../venv/bin/python -m unittest discover -s tests -t . -v"""
import logging

# Логи агента в тестах проверяются через assertLogs; в консоль их не выводим.
logging.getLogger("agent").addHandler(logging.NullHandler())
