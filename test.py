import deepspeech
import numpy as np
import librosa
import pandas as pd
from scipy.io import wavfile
from sklearn.metrics import accuracy_score
from difflib import SequenceMatcher  # 用于模糊比对

# 加载DeepSpeech模型
model_path = 'deepspeech-0.9.3-models.pbmm'
model = deepspeech.Model(model_path)

# 向音频中添加噪声的函数
def add_noise(audio_file, noise_level=0.05):
    signal, sr = librosa.load(audio_file, sr=None)
    noise = np.random.randn(len(signal))
    noisy_signal = signal + noise_level * noise
    return noisy_signal, sr

# 保存噪声音频的函数
def save_noisy_audio(noisy_signal, sr, output_file):
    wavfile.write(output_file, sr, np.int16(noisy_signal * 32767))  # 确保保存为 int16

# 转录音频的函数
def transcribe_audio(audio_file):
    # 加载音频文件并确保采样率为16kHz
    audio, sr = librosa.load(audio_file, sr=16000)

    # 将音频数据转换为 int16 类型
    audio = np.int16(audio * 32767)  # 将 float32 转为 int16

    # 执行转录
    text = model.stt(audio)
    return text

# 使用模糊比对计算准确度
def calculate_fuzzy_accuracy(predictions, labels):
    similarity_scores = []

    for pred, label in zip(predictions, labels):
        # 计算每对预测和真实标签的相似度
        similarity = SequenceMatcher(None, pred, label).ratio()
        similarity_scores.append(similarity)

    # 返回平均相似度作为模糊准确度
    return np.mean(similarity_scores)

# 自定义数据集，假设你的音频文件和标签如下
dataset = pd.read_csv('audio/label.csv', delimiter='\t')

# 主测试流程
def run_tests(dataset):
    predictions_clean = []
    predictions_noisy = []
    actual_labels = []

    for index, row in dataset.iterrows():
        audio_path = row['path']
        true_label = row['text']

        # 获取干净音频的转录结果
        clean_transcription = transcribe_audio("audio/pre/"+audio_path)
        predictions_clean.append(clean_transcription)

        # 向音频中添加噪声并保存
        noisy_audio, sr = add_noise(audio_path, noise_level=0.05)
        noisy_audio_path = f"audio/noisy/noisy_{audio_path}"
        save_noisy_audio(noisy_audio, sr, noisy_audio_path)

        # 获取噪声音频的转录结果
        noisy_transcription = transcribe_audio(noisy_audio_path)
        predictions_noisy.append(noisy_transcription)

        # 保存真实标签
        actual_labels.append(true_label)

    # 计算模糊准确度
    clean_fuzzy_accuracy = calculate_fuzzy_accuracy(predictions_clean, actual_labels)
    noisy_fuzzy_accuracy = calculate_fuzzy_accuracy(predictions_noisy, actual_labels)

    # 输出模糊准确度结果
    print(f"干净音频的模糊准确度: {clean_fuzzy_accuracy * 100:.2f}%")
    print(f"噪声音频的模糊准确度: {noisy_fuzzy_accuracy * 100:.2f}%")

    # 显示每个音频的转录结果与真实标签对比
    comparison_df = pd.DataFrame({
        'Audio File': dataset['path'],
        'True Label': actual_labels,
        'Clean Transcription': predictions_clean,
        'Noisy Transcription': predictions_noisy
    })

    print("\n转录结果对比:")
    print(comparison_df)

# 运行测试
run_tests(dataset)
