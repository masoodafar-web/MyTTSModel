# بهبودهای معماری MyTTSModel - نسخه Production-Ready

این سند توضیحات کامل بهبودهای معماری اعمال شده برای ارتقاء MyTTSModel به سطح production را ارائه می‌دهد.

## خلاصه بهبودها

در پاسخ به ایشو #[شماره]، بهبودهای زیر برای رفع ضعف‌های معماری و ارتقاء مدل به سطح production اعمال شده است:

### ✅ بهبودهای اعمال شده

#### 1. **Rotary Position Embedding (RoPE)** 
**وضعیت**: ✅ پیاده‌سازی شده

**مشکل قبلی**: استفاده از position encoding سنتی sinusoidal که relative position awareness ضعیفی دارد.

**راه‌حل**: 
- پیاده‌سازی RoPE به عنوان جایگزین مدرن
- قابلیت انتخاب بین `sinusoidal` و `rope` از طریق پارامتر `pos_encoding_type`
- بهبود آگاهی موقعیت نسبی در توکن‌ها

**مثال استفاده**:
```python
model = TransformerTTS(
    num_layers=8,
    d_model=512,
    num_heads=8,
    dff=2048,
    input_vocab_size=len(tokenizer),
    n_mels=80,
    pos_encoding_type='rope',  # استفاده از RoPE
    # ... سایر پارامترها
)
```

**مزایا**:
- بهبود relative position awareness
- عملکرد بهتر در سکانس‌های طولانی
- استفاده شده در مدل‌های مدرن مانند LLaMA و PaLM

**مرجع**: [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)

---

#### 2. **SwiGLU Activation در Feed-Forward Networks**
**وضعیت**: ✅ پیاده‌سازی شده

**مشکل قبلی**: استفاده از GELU ساده در FFN بدون بهره‌گیری از gating mechanism.

**راه‌حل**:
- پیاده‌سازی SwiGLU به عنوان جایگزین GELU
- قابلیت انتخاب بین `gelu` و `swiglu` از طریق پارامتر `activation`
- استفاده از gating mechanism برای یادگیری بهتر

**مثال استفاده**:
```python
model = TransformerTTS(
    num_layers=8,
    d_model=512,
    num_heads=8,
    dff=2048,
    input_vocab_size=len(tokenizer),
    n_mels=80,
    activation='swiglu',  # استفاده از SwiGLU
    # ... سایر پارامترها
)
```

**مزایا**:
- Gradient flow بهتر
- همگرایی سریع‌تر
- عملکرد بهتر در task‌های مختلف

**مرجع**: [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)

---

#### 3. **Temperature Sampling برای Inference**
**وضعیت**: ✅ پیاده‌سازی شده

**مشکل قبلی**: فقط greedy decoding ساده بدون کنترل تنوع خروجی.

**راه‌حل**:
- افزودن پارامتر `temperature` به متد `greedy_generate_fast`
- امکان کنترل تنوع و طبیعی بودن خروجی

**مثال استفاده**:
```python
# تولید deterministic (پیش‌فرض)
mel_hat, stop_probs = model.greedy_generate_fast(
    enc_ids,
    max_steps=600,
    temperature=1.0
)

# تولید با تنوع بیشتر
mel_hat, stop_probs = model.greedy_generate_fast(
    enc_ids,
    max_steps=600,
    temperature=1.2  # تنوع بیشتر
)
```

**مزایا**:
- کنترل بهتر بر کیفیت و تنوع خروجی
- امکان تنظیم trade-off بین دقت و طبیعی بودن

---

#### 4. **ماژول TTSMonitoring برای نظارت و دیباگ**
**وضعیت**: ✅ پیاده‌سازی شده

**مشکل قبلی**: ابزارهای محدود برای نظارت، دیباگ و تشخیص مشکلات alignment.

**راه‌حل**: ایجاد ماژول جامع `TTSMonitoring.py` با ابزارهای زیر:

##### 4.1. Alignment Visualization
```python
from TTSMonitoring import (
    extract_alignment_from_model,
    visualize_alignment_matrix,
    compute_alignment_diagonality
)

# استخراج alignment
alignment = extract_alignment_from_model(model, enc_ids, mel_target)

# محاسبه کیفیت alignment
diag_score = compute_alignment_diagonality(alignment)
print(f"Alignment quality: {diag_score:.4f}")

# تصویرسازی alignment
fig = visualize_alignment_matrix(
    alignment[0].numpy(),
    save_path="alignment.png"
)
```

##### 4.2. Gradient Analysis
```python
from TTSMonitoring import analyze_gradients

# تحلیل gradient برای تشخیص vanishing/exploding
grad_stats = analyze_gradients(model, loss, max_norm=1.0)
print(f"Gradient norms: min={grad_stats['min_norm']}, max={grad_stats['max_norm']}")

if "warning" in grad_stats:
    print(f"⚠️ {grad_stats['warning']}")
```

##### 4.3. Adaptive Gradient Clipping
```python
from TTSMonitoring import adaptive_gradient_clipping

grads = tape.gradient(loss, model.trainable_variables)

# استراتژی‌های مختلف clipping
clipped_grads, global_norm = adaptive_gradient_clipping(
    grads,
    max_norm=1.0,
    norm_type='percentile'  # یا 'global' یا 'per_layer'
)

optimizer.apply_gradients(zip(clipped_grads, model.trainable_variables))
```

##### 4.4. Metrics Aggregator
```python
from TTSMonitoring import MetricsAggregator

aggregator = MetricsAggregator(window_size=100)

for step in training_loop:
    aggregator.update({
        "loss": loss_value,
        "mel_error": mel_error,
        "stop_accuracy": stop_acc
    })
    
    if step % 100 == 0:
        stats = aggregator.get_all_statistics()
        for metric_name, metric_stats in stats.items():
            print(f"{metric_name}: mean={metric_stats['mean']:.4f}, trend={metric_stats['trend']}")
```

**مزایا**:
- تشخیص سریع مشکلات alignment
- نظارت بر gradient ها
- ردیابی metrics در طول زمان
- ابزارهای تصویرسازی برای دیباگ

---

#### 5. **پیکربندی‌های پیشرفته در TTSConfig**
**وضعیت**: ✅ به‌روزرسانی شده

بروزرسانی `ModelConfig` برای پشتیبانی از گزینه‌های جدید:

```python
@dataclass
class ModelConfig:
    num_layers: int
    d_model: int
    num_heads: int
    dff: int
    dropout_rate: float = 0.1
    droppath_rate: float = 0.05
    use_prenet: bool = True
    prenet_drop: float = 0.5
    cross_win: Optional[float] = 0.2
    max_length: int = 4096
    activation: str = 'gelu'  # 'gelu' یا 'swiglu' ✨ جدید
    pos_encoding_type: str = 'sinusoidal'  # 'sinusoidal' یا 'rope' ✨ جدید
```

---

## 🔄 سازگاری با نسخه‌های قبلی

**مهم**: تمام تغییرات به صورت backward-compatible پیاده‌سازی شده‌اند:

- پارامترهای جدید مقادیر پیش‌فرض دارند که رفتار قبلی را حفظ می‌کنند
- مدل‌های آموزش داده شده قبلی بدون تغییر قابل استفاده هستند
- API قدیمی بدون تغییر باقی مانده است

### مثال سازگاری
```python
# کد قدیمی (بدون تغییر کار می‌کند)
model = TransformerTTS(
    num_layers=8,
    d_model=512,
    num_heads=8,
    dff=2048,
    input_vocab_size=len(tokenizer),
    n_mels=80
)

# کد جدید با ویژگی‌های مدرن
model = TransformerTTS(
    num_layers=8,
    d_model=512,
    num_heads=8,
    dff=2048,
    input_vocab_size=len(tokenizer),
    n_mels=80,
    activation='swiglu',  # ✨ جدید
    pos_encoding_type='rope'  # ✨ جدید
)
```

---

## 📋 بهبودهای باقی‌مانده (برای آینده)

بر اساس ایشو، موارد زیر همچنان نیاز به پیاده‌سازی دارند:

### ⏳ اولویت بالا
1. **Duration Modeling**: افزودن duration predictor و length regulator (مانند FastSpeech2)
2. **Vocoder بهتر**: جایگزینی Griffin-Lim با HiFi-GAN یا UnivNet
3. **Monotonic Attention**: استفاده از MMA یا Location-sensitive attention

### ⏳ اولویت متوسط
4. **Multi-speaker Support**: افزودن speaker embeddings
5. **Text Normalization پیشرفته**: بهبود پردازش متن
6. **Beam Search**: پیاده‌سازی beam search برای inference

### ⏳ اولویت پایین
7. **Model Parallelism**: پشتیبانی از model parallelism برای مدل‌های بزرگ
8. **ZeRO/DeepSpeed**: یکپارچه‌سازی با ZeRO یا DeepSpeed
9. **Objective Metrics**: افزودن PESQ, STOI, DNSMOS

---

## 🎯 توصیه‌های استفاده

### برای آموزش مدل جدید
```python
from TTSConfig import get_model_preset

# استفاده از پیکربندی با ویژگی‌های مدرن
preset = get_model_preset('normal')
model = TransformerTTS(
    num_layers=preset.num_layers,
    d_model=preset.d_model,
    num_heads=preset.num_heads,
    dff=preset.dff,
    input_vocab_size=len(tokenizer),
    n_mels=80,
    dropout_rate=preset.dropout_rate,
    droppath_rate=preset.droppath_rate,
    activation='swiglu',  # توصیه برای مدل‌های جدید
    pos_encoding_type='rope',  # توصیه برای مدل‌های جدید
    use_prenet=preset.use_prenet,
    prenet_drop=preset.prenet_drop
)
```

### برای نظارت بر آموزش
```python
from TTSMonitoring import MetricsAggregator, analyze_gradients

aggregator = MetricsAggregator(window_size=100)

# در حلقه آموزش
for step, batch in enumerate(train_dataset):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, batch)
    
    grads = tape.gradient(loss, model.trainable_variables)
    
    # تحلیل gradient
    if step % 100 == 0:
        grad_stats = analyze_gradients(model, loss)
        print(f"Step {step}: {grad_stats}")
    
    # اعمال optimizer
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
    # ردیابی metrics
    aggregator.update({"loss": loss.numpy()})
```

### برای دیباگ مشکلات alignment
```python
from TTSMonitoring import extract_alignment_from_model, compute_alignment_diagonality

# در حین ارزیابی
for batch in val_dataset:
    alignment = extract_alignment_from_model(
        model, 
        batch['enc_ids'], 
        batch['mel_target']
    )
    
    score = compute_alignment_diagonality(alignment)
    if score < 0.5:
        print(f"⚠️ Poor alignment detected: {score:.4f}")
        # ذخیره برای بررسی
        visualize_alignment_matrix(
            alignment[0].numpy(),
            save_path=f"debug/alignment_step_{step}.png"
        )
```

---

## 📊 مقایسه قبل و بعد

| ویژگی | قبل | بعد |
|-------|-----|-----|
| Position Encoding | Sinusoidal ساده | Sinusoidal + RoPE ✨ |
| FFN Activation | GELU | GELU + SwiGLU ✨ |
| Inference Control | Greedy ساده | Greedy + Temperature ✨ |
| Gradient Clipping | Global norm | Global + Per-layer + Percentile ✨ |
| Monitoring | Metrics محدود | Alignment viz + Gradient analysis ✨ |
| Debugging Tools | خیر | TTSMonitoring module ✨ |

---

## 🔗 منابع و مراجع

1. **RoPE**: [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
2. **SwiGLU**: [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
3. **DropPath**: [Deep Residual Learning](https://arxiv.org/abs/1603.09382)
4. **Tacotron 2**: [Natural TTS Synthesis](https://arxiv.org/abs/1712.05884)
5. **Guided Attention**: [Efficiently Trainable TTS](https://arxiv.org/abs/1710.08969)

---

## 📝 نتیجه‌گیری

این بهبودها گام مهمی در جهت ارتقاء MyTTSModel به سطح production هستند. تمرکز بر:

✅ **معماری مدرن**: RoPE و SwiGLU برای عملکرد بهتر
✅ **کنترل بهتر**: Temperature sampling برای inference
✅ **نظارت جامع**: ابزارهای تصویرسازی و تحلیل
✅ **سازگاری**: حفظ backward compatibility

بهبودهای بعدی (Duration Modeling, Vocoder, Multi-speaker) در roadmap قرار دارند و در نسخه‌های آینده اضافه خواهند شد.

---

**تاریخ**: 2024  
**نسخه**: 1.0 - بهبودهای اولیه معماری
