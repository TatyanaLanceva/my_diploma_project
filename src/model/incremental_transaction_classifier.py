import pandas as pd
from pathlib import Path
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import Dataset
import numpy as np
import json
from datetime import datetime
from typing import List, Dict, Tuple
import logging
import random
import re

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === НАСТРОЙКИ ===
LABELING_FILE_XLSX = "operations_for_labeling.xlsx"
LABELING_FILE_CSV = "operations_for_labeling.xlsx - SHEET_NAME.csv"
MEMORY_FILE = "labeled_memory.pkl"
CORRECTIONS_FILE = "corrections_buffer.json"
MODEL_NAME = "DeepPavlov/rubert-base-cased"
OUTPUT_MODEL_DIR = Path("ruroberta_transaction_classifier")
EXACT_MATCHES_FILE = "exact_matches.json"

# Настройки инкрементального обучения
CORRECTION_THRESHOLD = 20
REPLAY_BUFFER_SIZE = 100
INCREMENTAL_LEARNING_RATE = 2e-5

# Создание папок
OUTPUT_MODEL_DIR.mkdir(exist_ok=True)


# ИЗМЕНЕНО: Функция для предобработки текста вынесена за пределы класса
def preprocess_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text).lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text


class TransactionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        if pd.isna(text) or text.strip() == "":
            text = "Unknown transaction"
            
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class CorrectionBuffer:
    def __init__(self, corrections_file: str = CORRECTIONS_FILE):
        self.corrections_file = corrections_file
        self.corrections = self._load_corrections()
        
    def _load_corrections(self) -> List[Dict]:
        if Path(self.corrections_file).exists():
            try:
                with open(self.corrections_file, 'r', encoding='utf-8') as f:
                    corrections = json.load(f)
                logger.info(f"🔄 Загружено {len(corrections)} исправлений из буфера")
                return corrections
            except Exception as e:
                logger.warning(f"⚠️ Ошибка загрузки исправлений: {e}. Буфер будет создан заново.")
        return []
    
    def _save_corrections(self):
        try:
            with open(self.corrections_file, 'w', encoding='utf-8') as f:
                json.dump(self.corrections, f, ensure_ascii=False, indent=2, default=str)
            logger.info(f"💾 Сохранено {len(self.corrections)} исправлений")
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения исправлений: {e}")
    
    def add_correction(self, transaction_text: str, predicted_category: str, 
                         correct_category: str, sber_category: str, operation_type: str, confidence: float = 0.0) -> bool:
        correction = {
            'transaction_text': transaction_text,
            'predicted_category': predicted_category,
            'correct_category': correct_category,
            'sber_category': sber_category,
            'operation_type': operation_type,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat(),
            'processed': False
        }
        
        self.corrections.append(correction)
        self._save_corrections()
        
        unprocessed_count = len([c for c in self.corrections if not c['processed']])
        logger.info(f"➕ Добавлено исправление. Необработанных: {unprocessed_count}/{CORRECTION_THRESHOLD}")
        
        return unprocessed_count >= CORRECTION_THRESHOLD
    
    def get_unprocessed_corrections(self) -> List[Dict]:
        return [c for c in self.corrections if not c['processed']]
    
    def mark_as_processed(self):
        for correction in self.corrections:
            if not correction['processed']:
                correction['processed'] = True
        self._save_corrections()
        logger.info("✅ Все исправления отмечены как обработанные")
    
    def get_stats(self) -> Dict:
        if not self.corrections:
            return {'total': 0, 'unprocessed': 0, 'categories': {}}
        
        unprocessed = self.get_unprocessed_corrections()
        category_stats = {}
        
        for correction in self.corrections:
            cat = correction['correct_category']
            if cat not in category_stats:
                category_stats[cat] = {'total': 0, 'unprocessed': 0}
            category_stats[cat]['total'] += 1
            if not correction['processed']:
                category_stats[cat]['unprocessed'] += 1
                
        return {
            'total': len(self.corrections),
            'unprocessed': len(unprocessed),
            'categories': category_stats
        }

class ExactMatchManager:
    def __init__(self, exact_matches_file: str = EXACT_MATCHES_FILE):
        self.exact_matches_file = exact_matches_file
        self.exact_matches = self._load_exact_matches()
    
    def _load_exact_matches(self) -> Dict[str, str]:
        if Path(self.exact_matches_file).exists():
            try:
                with open(self.exact_matches_file, 'r', encoding='utf-8') as f:
                    matches = json.load(f)
                logger.info(f"🎯 Загружено {len(matches)} точных совпадений")
            except Exception as e:
                logger.warning(f"⚠️ Ошибка загрузки точных совпадений: {e}. Будет создан новый файл.")
            return {}
        return {}
    
    def _save_exact_matches(self):
        try:
            with open(self.exact_matches_file, 'w', encoding='utf-8') as f:
                json.dump(self.exact_matches, f, ensure_ascii=False, indent=2)
            logger.info(f"💾 Сохранено {len(self.exact_matches)} точных совпадений")
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения точных совпадений: {e}")
    
    def add_exact_match(self, combined_text: str, category: str):
        normalized_text = combined_text.lower().strip()
        self.exact_matches[normalized_text] = category
        self._save_exact_matches()
    
    def get_exact_match(self, combined_text: str) -> str:
        normalized_text = combined_text.lower().strip()
        return self.exact_matches.get(normalized_text, None)

class CustomTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if self.class_weights is not None:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            
        return (loss, outputs) if return_outputs else loss

class IncrementalTransactionClassifier:
    def __init__(self, model_dir: str = str(OUTPUT_MODEL_DIR)):
        self.model_dir = Path(model_dir)
        self.correction_buffer = CorrectionBuffer()
        self.exact_match_manager = ExactMatchManager()
        
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        
        self._load_or_create_model()
    
    def _load_or_create_model(self):
        try:
            if (self.model_dir / "config.json").exists() and (self.model_dir / "label_encoder.pkl").exists():
                logger.info(f"📂 Загружаем существующую модель из {self.model_dir}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
                self.label_encoder = joblib.load(self.model_dir / "label_encoder.pkl")
                logger.info("✅ Модель загружена успешно")
            else:
                logger.info("🆕 Файлы модели не найдены. Создаем новую модель.")
                self._create_new_model()
        except Exception as e:
            logger.error(f"❌ Ошибка при загрузке модели: {e}")
            logger.info("🔄 Создаем новую модель")
            self._create_new_model()
            
    def _create_new_model(self):
        logger.info("🚀 Создание новой модели...")
        
        df_memory = load_initial_data()
        if df_memory is None or df_memory.empty:
            logger.error("❌ Нет данных для создания модели")
            return
            
        df_memory = self._clean_data(df_memory)
        
        if df_memory.empty:
            logger.error("❌ Нет корректных данных для обучения")
            return
        
        self.label_encoder = LabelEncoder()
        df_memory['label'] = self.label_encoder.fit_transform(df_memory['custom_category'])
        
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=len(self.label_encoder.classes_)
        )
        
        self._train_model(df_memory)
        self._save_model()
        
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.dropna(subset=['description', 'custom_category'], inplace=True)
        df = df[df['custom_category'].str.strip() != '']
        
        df['sber_category'] = df['sber_category'].astype(str).fillna('Нет категории').str.strip()
        df['operation_type'] = df['operation_type'].astype(str).fillna('Нет типа').str.strip()
        df['description'] = df['description'].astype(str).str.strip()
        df['custom_category'] = df['custom_category'].astype(str).str.strip()
        
        df = df[df['description'] != '']
        df = df[df['description'] != 'nan']
        
        df['combined_text'] = (
            df['description'].apply(preprocess_text) +
            ' | SBER: ' + df['sber_category'].apply(preprocess_text) +
            ' | ТИП: ' + df['operation_type'].apply(preprocess_text)
        )
        
        class_counts = df['custom_category'].value_counts()
        classes_to_keep = class_counts[class_counts >= 2].index
        df = df[df['custom_category'].isin(classes_to_keep)]
        
        return df.reset_index(drop=True)
    
    def _train_model(self, df: pd.DataFrame, is_incremental: bool = False):
        X = df['combined_text'].tolist()
        y = df['label'].values
        
        if len(y) < 2:
            logger.warning("Недостаточно данных для обучения.")
            return

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        train_dataset = TransactionDataset(X_train, y_train, self.tokenizer)
        val_dataset = TransactionDataset(X_val, y_val, self.tokenizer)

        class_counts = pd.Series(y_train).value_counts()
        total_samples = len(y_train)
        class_weights = total_samples / (len(class_counts) * class_counts)
        weights_tensor = torch.tensor(class_weights.loc[range(len(self.label_encoder.classes_))].values, dtype=torch.float)
        
        num_epochs = 20 if not is_incremental else 2
        learning_rate = INCREMENTAL_LEARNING_RATE if is_incremental else 3e-5
        
        training_args = TrainingArguments(
            output_dir=self.model_dir / "temp",
            num_train_epochs=num_epochs,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            eval_strategy="epoch",
            save_strategy="no",
            logging_steps=20,
            dataloader_num_workers=0,
            learning_rate=learning_rate,
            warmup_ratio=0.1,
            load_best_model_at_end=False,
            metric_for_best_model='f1'
        )
        
        def compute_metrics(pred):
            labels = pred.label_ids
            preds = pred.predictions.argmax(-1)
            return {
                'accuracy': accuracy_score(labels, preds),
                'f1': f1_score(labels, preds, average='weighted')
            }
        
        trainer = CustomTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            class_weights=weights_tensor
        )
        
        if torch.cuda.is_available():
            self.model.to('cuda')
        
        logger.info(f"⏳ {'Инкрементальное' if is_incremental else 'Полное'} обучение...")
        trainer.train()
        
        eval_result = trainer.evaluate()
        logger.info(f"📈 Accuracy: {eval_result['eval_accuracy']:.4f}, F1: {eval_result['eval_f1']:.4f}")
    
    def _save_model(self):
        try:
            self.model.save_pretrained(self.model_dir)
            self.tokenizer.save_pretrained(self.model_dir)
            joblib.dump(self.label_encoder, self.model_dir / "label_encoder.pkl")
            logger.info(f"💾 Модель сохранена в {self.model_dir}")
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения модели: {e}")
    
    # ИЗМЕНЕНО: Добавляем возвращаемый список категорий
    def predict_and_get_categories(self, transaction_text: str, sber_category: str = "", operation_type: str = "") -> Tuple[str, float, List[str]]:
        combined_text = (
            preprocess_text(transaction_text) +
            ' | SBER: ' + preprocess_text(sber_category) +
            ' | ТИП: ' + preprocess_text(operation_type)
        )
        
        # Сначала проверяем точные совпадения
        exact_match = self.exact_match_manager.get_exact_match(combined_text)
        if exact_match:
            logger.info(f"🎯 Точное совпадение найдено: {exact_match}")
            return exact_match, 1.0, list(self.label_encoder.classes_)
        
        if not self.model or not self.tokenizer or not self.label_encoder:
            logger.error("❌ Модель не загружена")
            return "Неизвестно", 0.0, []
        
        try:
            encoding = self.tokenizer(
                combined_text,
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors='pt'
            )
            
            with torch.no_grad():
                outputs = self.model(**encoding)
                probabilities = torch.softmax(outputs.logits, dim=-1)
                confidence, predicted_class = torch.max(probabilities, dim=1)
            
            predicted_category = self.label_encoder.inverse_transform([predicted_class.item()])[0]
            confidence_score = confidence.item()
            
            return predicted_category, confidence_score, list(self.label_encoder.classes_)
            
        except Exception as e:
            logger.error(f"❌ Ошибка предсказания: {e}")
            return "Ошибка", 0.0, []
    
    def add_correction(self, transaction_text: str, predicted_category: str, 
                         correct_category: str, sber_category: str, operation_type: str, confidence: float = 0.0) -> Dict[str, str]:
        combined_text = (
            preprocess_text(transaction_text) +
            ' | SBER: ' + preprocess_text(sber_category) +
            ' | ТИП: ' + preprocess_text(operation_type)
        )
        
        self.exact_match_manager.add_exact_match(combined_text, correct_category)
        
        should_retrain = self.correction_buffer.add_correction(
            transaction_text, predicted_category, correct_category, sber_category, operation_type, confidence
        )
        
        result = {
            'status': 'added',
            'message': f'Исправление добавлено. Точное совпадение сохранено.'
        }
        
        if should_retrain:
            retrain_result = self.incremental_retrain()
            result['retrain_status'] = retrain_result['status']
            result['message'] += f" {retrain_result['message']}"
        
        return result
    
    def incremental_retrain(self) -> Dict[str, str]:
        try:
            corrections = self.correction_buffer.get_unprocessed_corrections()
            if not corrections:
                return {'status': 'no_data', 'message': 'Нет исправлений для обучения'}
            
            logger.info(f"🔄 Начинаем инкрементальное обучение на {len(corrections)} исправлениях")
            
            correction_df = pd.DataFrame(corrections)
            correction_df = self._clean_data(correction_df.rename(columns={'transaction_text': 'description', 'correct_category': 'custom_category'}))

            old_data = self._get_replay_buffer(len(corrections))
            
            all_data = old_data + correction_df.to_dict('records')
            
            train_df = pd.DataFrame(all_data)

            existing_categories = set(self.label_encoder.classes_)
            new_categories = set(train_df['custom_category']) - existing_categories
            
            if new_categories:
                logger.info(f"🆕 Найдены новые категории: {new_categories}")
                all_known_categories = list(existing_categories) + list(new_categories)
                self.label_encoder = LabelEncoder()
                self.label_encoder.fit(all_known_categories)
                
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    MODEL_NAME,
                    num_labels=len(self.label_encoder.classes_)
                )
            
            train_df['label'] = self.label_encoder.transform(train_df['custom_category'])
            
            self._train_model(train_df, is_incremental=True)
            self._save_model()
            
            self.correction_buffer.mark_as_processed()
            
            return {
                'status': 'success', 
                'message': f'Модель дообучена на {len(corrections)} исправлениях + {len(old_data)} старых примерах'
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка инкрементального обучения: {e}")
            return {'status': 'error', 'message': f'Ошибка: {str(e)}'}
    
    def _get_replay_buffer(self, correction_count: int) -> List[Dict]:
        memory = load_memory()
        if not memory:
            return []
        
        sample_size = min(REPLAY_BUFFER_SIZE, len(memory), correction_count * 3)
        
        if sample_size < len(memory):
            memory = random.sample(memory, sample_size)
        
        logger.info(f"🔄 Используем {len(memory)} старых примеров для replay buffer")
        return memory
    
    def get_stats(self) -> Dict:
        correction_stats = self.correction_buffer.get_stats()
        exact_matches_count = len(self.exact_match_manager.exact_matches)
        
        model_info = {
            'model_loaded': self.model is not None,
            'num_classes': len(self.label_encoder.classes_) if self.label_encoder else 0,
            'classes': list(self.label_encoder.classes_) if self.label_encoder else []
        }
        
        return {
            'corrections': correction_stats,
            'exact_matches': exact_matches_count,
            'model': model_info
        }

# === ФУНКЦИИ ДЛЯ ЗАГРУЗКИ ДАННЫХ ===
def load_initial_data() -> pd.DataFrame:
    try:
        if Path(LABELING_FILE_XLSX).exists():
            logger.info(f"📊 Загружаем данные из {LABELING_FILE_XLSX}")
            return pd.read_excel(LABELING_FILE_XLSX)
        elif Path(LABELING_FILE_CSV).exists():
            logger.info(f"📊 Загружаем данные из {LABELING_FILE_CSV}")
            return pd.read_csv(LABELING_FILE_CSV)
        else:
            logger.error(f"❌ Файл с разметкой не найден: {LABELING_FILE_XLSX} или {LABELING_FILE_CSV}")
            return None
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки файла: {e}")
        return None

def load_memory():
    if Path(MEMORY_FILE).exists():
        try:
            memory = joblib.load(MEMORY_FILE)
            logger.info(f"🧠 Загружено {len(memory)} записей из памяти")
            return memory
        except Exception as e:
            logger.warning(f"⚠️ Ошибка загрузки памяти: {e}")
    
    initial_data = load_initial_data()
    if initial_data is not None:
        memory = initial_data.to_dict('records')
        save_memory(memory)
        return memory
    
    return []

def save_memory(memory):
    try:
        joblib.dump(memory, MEMORY_FILE)
        logger.info(f"💾 Сохранено {len(memory)} записей в памяти")
    except Exception as e:
        logger.error(f"❌ Ошибка сохранения памяти: {e}")

# === ОСНОВНАЯ ФУНКЦИЯ ===
def main():
    print("🚀 Система инкрементального обучения для классификации транзакций")
    print("======================================================================")
    
    classifier = IncrementalTransactionClassifier()
    
    stats = classifier.get_stats()
    print(f"📊 Статистика системы:")
    print(f"   - Модель загружена: {stats['model']['model_loaded']}")
    print(f"   - Количество классов: {stats['model']['num_classes']}")
    print(f"   - Исправлений в буфере: {stats['corrections']['total']} (необработанных: {stats['corrections']['unprocessed']})")
    print(f"   - Точных совпадений: {stats['exact_matches']}")
    
    print("\n💡 Для выхода из интерактивного режима нажмите Ctrl+C.")
    while True:
        try:
            # === Интерактивный ввод данных для предсказания ===
            print("\n" + "="*50)
            trans_text = input("Введите описание транзакции (например, 'Покупка в Магнит'): ")
            sber_cat = input("Введите категорию от Сбербанка (например, 'Продукты питания'): ")
            op_type = input("Введите тип операции (например, 'Расход'): ")
            
            # === Получение предсказания и списка категорий ===
            predicted_cat, confidence, all_categories = classifier.predict_and_get_categories(
                trans_text, sber_cat, op_type
            )
            
            print(f"\n🔍 Предсказание модели:")
            print(f"   Предполагаемая категория: {predicted_cat}")
            print(f"   Уверенность: {confidence:.4f}")
            
            # === Запрос исправления у пользователя ===
            if confidence < 0.8: # Порог для запроса исправления
                print("\n🤔 Модель не уверена. Подскажите правильную категорию.")
                print("Список категорий:")
                # Выводим категории с номерами для удобного выбора
                for i, cat in enumerate(sorted(all_categories)):
                    print(f"   {i+1}: {cat}")
                
                print(f"\n   0: '{predicted_cat}' (Подтвердить предложенное)")
                user_choice = input("Введите номер правильной категории или '0' для подтверждения: ")
                
                if user_choice.isdigit():
                    choice_index = int(user_choice)
                    if 1 <= choice_index <= len(all_categories):
                        correct_category = sorted(all_categories)[choice_index - 1]
                        print(f"✅ Вы выбрали: {correct_category}")
                        
                        result = classifier.add_correction(
                            transaction_text=trans_text,
                            predicted_category=predicted_cat,
                            correct_category=correct_category,
                            sber_category=sber_cat,
                            operation_type=op_type,
                            confidence=confidence
                        )
                        print(f"   Результат: {result['message']}")
                    elif choice_index == 0:
                        print("✅ Предложенная категория подтверждена.")
                    else:
                        print("⚠️ Некорректный выбор. Попробуйте еще раз.")
                else:
                    print("⚠️ Некорректный ввод. Пожалуйста, введите число.")
            else:
                print("\n✅ Модель достаточно уверена в предсказании. Исправление не требуется.")
        
        except KeyboardInterrupt:
            print("\n\n👋 Интерактивный режим завершен. До свидания!")
            break
        except Exception as e:
            logger.error(f"❌ Произошла непредвиденная ошибка: {e}")
            print("\n❌ Произошла ошибка. Пожалуйста, попробуйте еще раз.")

if __name__ == "__main__":
    main()