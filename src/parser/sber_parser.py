# src/parser/sber_parser.py

import pdfplumber
import re
import pandas as pd

# Категории из Сбербанка
categories = [
    'Супермаркеты', 'Перевод на карту', 'Прочие расходы', 'Рестораны и кафе',
    'Коммунальные платежи, связь, интернет.', 'Прочие операции', 'Отдых и развлечения',
    'Оплата с нескольких счетов', 'Зачисление', 'Возврат, отмена операции',
    'Одежда и аксессуары', 'Транспорт', 'Здоровье и красота', 'Все для дома', 'Автомобиль'
]
escaped_categories = [re.escape(cat) for cat in categories]
category_pattern = re.compile(r'\b(' + '|'.join(escaped_categories) + r')\b')

def parse_sber_statement(pdf_path):
    """
    Парсит PDF-выписку Сбербанка и возвращает DataFrame с операциями.
    """
    transactions = []
    with pdfplumber.open(pdf_path) as pdf:
        full_text = ""
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"

    lines = [line.strip() for line in full_text.split('\n') if line.strip()]

    i = 0
    while i < len(lines):
        line = lines[i]

        if any(keyword in line for keyword in [
            "Выписка по счёту", "КРЕДИТНАЯ СБЕРКАРТА", "Продолжение на следующей странице",
            "Страница", "ДАТА ОПЕРАЦИИ", "Сумма в валюте", "ОСТАТОК СРЕДСТВ"
        ]):
            i += 1
            continue

        datetime_match = re.match(r'(\d{2}\.\d{2}\.\d{4}) \d{2}:\d{2}', line)
        if not datetime_match:
            i += 1
            continue

        date_str = datetime_match.group(1)
        rest_line = line[len(date_str):].strip()

        time_end = rest_line.find(' ')
        if time_end == -1:
            i += 1
            continue
        rest_line = rest_line[time_end:].strip()

        cat_match = category_pattern.search(rest_line)
        if not cat_match:
            i += 1
            continue

        category = cat_match.group(1)
        after_cat = rest_line[cat_match.end():].strip()

        ab_match = re.search(r'([+-]?\d[\d\s]*,\d{2})\s+([\d\s,]+)', after_cat)
        if not ab_match:
            i += 1
            continue

        amount_raw = ab_match.group(1).replace(' ', '').replace(',', '.')
        amount = abs(float(amount_raw))
        sign = '+' if '+' in amount_raw else '-'
        desc_end = ab_match.start()
        description_part = after_cat[:desc_end].strip()
        full_desc = description_part

        while i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            if next_line and not re.match(r'\d{2}\.\d{2}\.\d{4} \d{2}:\d{2}', next_line):
                full_desc += " " + next_line
                i += 1
            else:
                break

        full_desc_lower = full_desc.lower()
        if category == "Перевод на карту" and "перевод от л. татьяна юрьевна" in full_desc_lower:
            flow_type = "transfer_in"
            is_real = False
            op_type = "Перемещение"
        elif category == "Оплата с нескольких счетов" and "пополнение карты" in full_desc_lower:
            flow_type = "transfer_in"
            is_real = False
            op_type = "Перемещение"
        elif category == "Зачисление" and "пополнение счета" in full_desc_lower:
            flow_type = "transfer_in"
            is_real = False
            op_type = "Перемещение"
        elif category == "Возврат, отмена операции":
            flow_type = "refund"
            is_real = True
            op_type = "Возврат"
        else:
            flow_type = "real"
            is_real = True
            op_type = "Доход" if sign == '+' else "Расход"

        transactions.append({
            'date': date_str,
            'category': category,
            'amount': amount,
            'sign': sign,
            'description': full_desc,
            'type': op_type,
            'flow_type': flow_type,
            'is_real': is_real,
            'source_file': pdf_path.name
        })
        i += 1

    return pd.DataFrame(transactions)