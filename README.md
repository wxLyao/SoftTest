#  软件测试代码大作业

T38  组长: 周宇健 

| 学号      | 姓名   |
| --------- | ------ |
| 221250136 | 周宇健 |
| 221250081 | 王鹏龙 |
| 221250030 | 干明瑞 |

## 实验说明

### 1. 代码大作业选题

面向XXX场景的深度学习模型测试技术

### 2. 实验简介

本实验旨在测试 DeepSpeech 模型在噪声环境下的文本识别能力，评估其抗干扰性能。实验通过对干净和添加噪声的音频进行语音转录，比较其识别的文本准确度。模糊匹配准确度（Fuzzy Accuracy）作为衡量指标，表示预测文本与真实文本之间的相似程度。实验数据集包括音频文件及其对应的真实文本标签，噪声以高斯噪声的形式添加到音频信号中。

### 3.代码分析

#### **1. 模型加载**

```python
model_path = 'deepspeech-0.9.3-models.pbmm'
model = deepspeech.Model(model_path)
```

- **功能**: 使用 DeepSpeech 提供的预训练模型进行语音转录。
- **分析**: 模型路径需根据实际文件位置设置，建议在代码开头添加模型文件检查逻辑，以确保路径有效。

------

#### **2. 添加噪声**

```python
def add_noise(audio_file, noise_level=0.05):
    signal, sr = librosa.load(audio_file, sr=None)
    noise = np.random.randn(len(signal))
    noisy_signal = signal + noise_level * noise
    return noisy_signal, sr
```

- **功能**: 加载音频文件，生成高斯噪声并叠加在音频信号上。

------

#### **3. 保存噪声音频**

```python
def save_noisy_audio(noisy_signal, sr, output_file):
    wavfile.write(output_file, sr, np.int16(noisy_signal * 32767))
```

- **功能**: 将生成的带噪音频保存为 `.wav` 文件。

------

#### **4. 转录音频**

```py
def transcribe_audio(audio_file):
    audio, sr = librosa.load(audio_file, sr=16000)
    audio = np.int16(audio * 32767)
    text = model.stt(audio)
    return text
```

- **功能**: 加载音频并通过 DeepSpeech 模型进行转录。

------

#### **5. 模糊准确度计算**

```py
def calculate_fuzzy_accuracy(predictions, labels):
    similarity_scores = []
    for pred, label in zip(predictions, labels):
        similarity = SequenceMatcher(None, pred, label).ratio()
        similarity_scores.append(similarity)
    return np.mean(similarity_scores)
```

- **功能**: 使用 `SequenceMatcher` 计算预测文本与真实文本的相似度。

------

#### **6. 主测试流程**

```py
def run_tests(dataset):
    predictions_clean = []
    predictions_noisy = []
    actual_labels = []

    for index, row in dataset.iterrows():
        audio_path = row['path']
        true_label = row['text']

        clean_transcription = transcribe_audio("audio/pre/"+audio_path)
        predictions_clean.append(clean_transcription)

        noisy_audio, sr = add_noise("audio/pre/"+audio_path, noise_level=0.05)
        noisy_audio_path = f"audio/noisy/noisy_{audio_path}"
        save_noisy_audio(noisy_audio, sr, noisy_audio_path)

        noisy_transcription = transcribe_audio(noisy_audio_path)
        predictions_noisy.append(noisy_transcription)

        actual_labels.append(true_label)

    clean_fuzzy_accuracy = calculate_fuzzy_accuracy(predictions_clean, actual_labels)
    noisy_fuzzy_accuracy = calculate_fuzzy_accuracy(predictions_noisy, actual_labels)

    print(f"干净音频的模糊准确度: {clean_fuzzy_accuracy * 100:.2f}%")
    print(f"噪声音频的模糊准确度: {noisy_fuzzy_accuracy * 100:.2f}%")

    comparison_df = pd.DataFrame({
        'Audio File': dataset['path'],
        'True Label': actual_labels,
        'Clean Transcription': predictions_clean,
        'Noisy Transcription': predictions_noisy
    })

    print("\n转录结果对比:")
    print(comparison_df)
```

- **功能**: 测试 DeepSpeech 在干净与噪声音频上的转录性能，输出模糊准确度和转录结果。

### 4.实验过程

#### 1. **数据准备**

- 数据集采用 CSV 格式，包含音频文件路径和对应的真实转录文本。

#### 2. **模型加载**

- 使用 `deepspeech.Model` 加载 DeepSpeech 预训练模型。
- 模型文件路径：`deepspeech-0.9.3-models.pbmm`。

#### 3. **音频处理**

- 干净音频直接转录。
- 添加噪声音频处理：
  1. 从音频文件中加载信号。
  2. 生成与信号等长的高斯噪声，按比例叠加至信号。
  3. 保存处理后的噪声音频供转录测试。

#### 4. **文本转录与对比**

- 使用 DeepSpeech 对音频进行转录，获取模型输出文本。
- 比较干净音频和噪声音频的转录文本，与真实标签进行模糊准确度计算。

#### 5. **准确度评估**

- 通过模糊匹配计算准确度：
  - 干净音频的模糊准确度。
  - 噪声音频的模糊准确度。
- 输出每条记录的详细结果对比。

------

### **实验结果**

- ![结果展示](assets\结果展示.png)

- 实验结果如上：

  匹配文本模糊准确度：

  - 干净音频：94.22%
  - 噪声音频：72.32%

------

### **实验结论**

- 当前模型抗噪能力一般。
- 模型优化方向：数据增强（加入噪声样本训练）和更强的抗噪设计。

## 代码运行说明

### 1.环境配置

与项目根目录执行以下命令

```
# Create and activate a virtualenv
virtualenv -p python3 $HOME/tmp/deepspeech-venv/
source $HOME/tmp/deepspeech-venv/bin/activate

# Install DeepSpeech
pip3 install deepspeech

# Download pre-trained English model files
curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.pbmm
```

### 2.运行 

运行test.py文件

运行结果展示于Untitled.ipynb文件

![结果展示](assets\结果展示.png)

