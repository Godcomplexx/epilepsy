# Система предсказания эпилептических приступов

Персонализированное предсказание эпилептических приступов с использованием Transfer Learning на данных ЭЭГ из базы CHB-MIT.

## Обзор

Система предсказывает эпилептические приступы с помощью архитектуры **CNN-LSTM** и transfer learning:
1. **Предобучение** глобальной модели на данных нескольких пациентов
2. **Дообучение** (fine-tuning) для каждого пациента индивидуально
3. **Персонализированные пороги** для оптимального баланса чувствительности и специфичности

### Ключевые особенности
- **Transfer Learning**: Глобальное предобучение + персональное дообучение
- **Адаптивный Learning Rate**: Автоматическая настройка в зависимости от объёма данных пациента
- **Focal Loss**: Улучшенная работа с дисбалансом классов (preictal << interictal)
- **Отбраковка артефактов**: Автоматическое удаление зашумлённых окон
- **Персонализированные пороги**: Поиск оптимального порога для каждого пациента

## Структура проекта

```
epilepsy/
├── config/
│   └── default.yaml          # Параметры конфигурации
├── src/
│   ├── data/
│   │   ├── index_builder.py  # Построение индексов приступов/файлов из CHB-MIT
│   │   ├── labeling.py       # Разметка окон (preictal/interictal)
│   │   ├── preprocessing.py  # Загрузка EDF, фильтрация, ресемплинг
│   │   └── segmentation.py   # Сегментация скользящим окном
│   ├── features/
│   │   └── extractor.py      # Извлечение признаков: PSD, статистика, Hjorth, энтропия
│   ├── models/
│   │   ├── classifier.py     # Классические ML-классификаторы (RF, XGB, SVM)
│   │   └── deep_model.py     # CNN-LSTM с attention, Focal Loss, MAML
│   ├── evaluation/
│   │   ├── alarm_logic.py    # Генерация тревог со сглаживанием
│   │   └── metrics.py        # Метрики: Sensitivity, FA/24h, event-based
│   ├── pipeline.py           # Пайплайн классического ML
│   └── pipeline_transfer.py  # Пайплайн Transfer Learning
├── run.py                    # Запуск классического ML пайплайна
├── run_transfer.py           # Запуск Transfer Learning пайплайна
└── requirements.txt          # Python зависимости
```

## Установка

```bash
# Создание виртуального окружения
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Установка зависимостей
pip install -r requirements.txt

# Для поддержки GPU (рекомендуется)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Данные

Проект использует базу данных **CHB-MIT Scalp EEG Database**:
- 24 пациента с фармакорезистентной эпилепсией
- 844 часа непрерывных записей ЭЭГ
- 198 размеченных приступов

Скачать: https://physionet.org/content/chbmit/1.0.0/

Укажите путь к данным в `config/default.yaml`:
```yaml
paths:
  data_root: "путь/к/chbmit/1.0.0"
```

## Использование

### Transfer Learning пайплайн (рекомендуется)

```bash
# Полный пайплайн с разделением train/test
python run_transfer.py --config config/default.yaml

# Только определённые пациенты
python run_transfer.py --patients chb01 chb02 chb03

# Продолжить с предобученной модели
python run_transfer.py --resume
```

### Классический ML пайплайн

```bash
python run.py --config config/default.yaml --patients chb01 chb02
```

## Конфигурация

Ключевые параметры в `config/default.yaml`:

| Параметр | По умолчанию | Описание |
|----------|--------------|----------|
| `timing.sph` | 60 сек | Seizure Prediction Horizon — минимальный запас до начала приступа |
| `timing.preictal_duration` | 1800 сек | Длительность preictal периода (30 мин) |
| `windowing.window_length` | 4 сек | Размер окна ЭЭГ |
| `deep_learning.pretrain_epochs` | 30 | Эпохи предобучения глобальной модели |
| `deep_learning.finetune_epochs` | 20 | Эпохи дообучения на пациенте |
| `deep_learning.use_focal_loss` | true | Использовать Focal Loss для дисбаланса классов |

## Архитектура модели

```
Вход: (batch, 17 каналов, 1024 отсчёта)
    ↓
CNN Block 1: Conv1d(17→32) + BatchNorm + ReLU + MaxPool
CNN Block 2: Conv1d(32→64) + BatchNorm + ReLU + MaxPool  
CNN Block 3: Conv1d(64→128) + BatchNorm + ReLU + MaxPool
    ↓
LSTM: 2 слоя, hidden=64, bidirectional
    ↓
Attention: Self-attention по временным шагам
    ↓
FC: 64 → 1 (sigmoid)
    ↓
Выход: P(preictal)
```

## Результаты

### Последний запуск (24 пациента)

| Группа | Пациентов | Приступов | Обнаружено | Sensitivity | Mean FA/24h |
|--------|-----------|-----------|------------|-------------|-------------|
| TRAIN | 19 | 182 | 117 | 64.3% | 45.27 |
| TEST | 5 | 16 | 13 | **81.2%** | 34.31 |

**Лучшие результаты:**
- chb02: Sensitivity=100%, FA/24h=0.00, AUC=0.877
- chb10: Sensitivity=100%, FA/24h=0.00, AUC=0.777
- chb11 (TEST): Sensitivity=100%, FA/24h=2.92, AUC=0.776

## Метрики оценки

- **Sensitivity (чувствительность)**: % приступов с хотя бы одной тревогой в окне предсказания
- **FA/24h**: Ложные тревоги за 24 часа записи
- **AUC**: Площадь под ROC-кривой (классификация на уровне окон)

Окно предсказания: `[onset - SOP, onset - SPH]` = `[onset - 10мин, onset - 1мин]`

## Известные проблемы

Некоторые пациенты показывают низкие результаты из-за:
- **Атипичные паттерны приступов** (chb15: 0% sensitivity при 20 приступах)
- **Недостаточно данных** (chb18: только 1 файл)
- **Высокая вариабельность между приступами** (chb12: 40 приступов, 47.5% sensitivity)

## Планы по улучшению

- [ ] Multi-task learning между пациентами
- [ ] Классификация типов приступов
- [ ] Online learning / адаптация в реальном времени
- [ ] Снижение FA/24h с помощью постобработки
- [ ] Добавление энтропийных признаков

## Ссылки

1. CHB-MIT Database: Shoeb, A. H. (2009). Application of machine learning to epileptic seizure onset detection and treatment.
2. Focal Loss: Lin, T. Y., et al. (2017). Focal loss for dense object detection.

## Лицензия

MIT License
