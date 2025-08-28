import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import jieba
import re

# 1. 数据加载
def load_data(filepath):
    try:
        data = pd.read_csv(filepath)
        print("数据加载成功!")
        print(f"数据形状: {data.shape}")
        print("\n数据预览:")
        print(data.head())
        return data
    except Exception as e:
        print(f"数据加载失败: {e}")
        return None

# 2. 数据预处理
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    # 去除特殊字符和数字
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    # 分词
    words = jieba.lcut(text)
    # 去除停用词
    stopwords = ["的", "了", "和", "是", "就", "都", "也", "在", "与", "等"]
    words = [word for word in words if word not in stopwords]
    return ' '.join(words)

def preprocess_data(data):
    # 检查数据是否有缺失
    print("\n缺失值统计:")
    print(data.isnull().sum())
    
    # 假设数据有'text'和'label'列，根据实际情况调整
    if 'text' not in data.columns or 'label' not in data.columns:
        raise ValueError("数据必须包含'text'和'label'列")
    
    # 填充可能的缺失值
    data['text'] = data['text'].fillna('')
    
    # 文本预处理
    print("\n正在进行文本预处理...")
    data['processed_text'] = data['text'].apply(preprocess_text)
    
    # 查看标签分布
    print("\n标签分布:")
    print(data['label'].value_counts())
    
    return data

# 3. 特征提取
def extract_features(data):
    # 使用TF-IDF向量化文本
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = tfidf.fit_transform(data['processed_text'])
    y = data['label']
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, tfidf

# 4. 模型训练与评估
def train_and_evaluate(X_train, X_test, y_train, y_test):
    models = {
        "Naive Bayes": MultinomialNB(),
        "Linear SVM": LinearSVC(),
        "Random Forest": RandomForestClassifier(n_estimators=100)
    }
    
    results = {}
    for name, model in models.items():
        print(f"\n训练 {name} 模型...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        results[name] = {
            "model": model,
            "accuracy": acc,
            "report": report
        }
        
        print(f"{name} 准确率: {acc:.4f}")
        print("分类报告:")
        print(report)
    
    return results

# 5. 可视化分析
def visualize_results(data, results, tfidf):
    # 标签分布可视化
    plt.figure(figsize=(10, 5))
    sns.countplot(x='label', data=data)
    plt.title('标签分布')
    plt.show()
    
    # 模型性能比较
    model_names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in model_names]
    
    plt.figure(figsize=(10, 5))
    sns.barplot(x=model_names, y=accuracies)
    plt.title('模型准确率比较')
    plt.ylim(0, 1)
    plt.show()
    
    # 词云可视化
    text = ' '.join(data['processed_text'])
    wordcloud = WordCloud(
        font_path='simhei.ttf',
        width=800,
        height=600,
        background_color='white'
    ).generate(text)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('评论词云')
    plt.show()

# 主函数
def main():
    # 文件路径 - 请根据实际情况修改
    filepath = "D:/ai/week07/第七周 文本分类/week7 文本分类问题/week7 文本分类问题/文本分类练习.csv"
    
    # 1. 加载数据
    data = load_data(filepath)
    if data is None:
        return
    
    # 2. 数据预处理
    try:
        data = preprocess_data(data)
    except Exception as e:
        print(f"数据预处理错误: {e}")
        return
    
    # 3. 特征提取
    X_train, X_test, y_train, y_test, tfidf = extract_features(data)
    
    # 4. 模型训练与评估
    results = train_and_evaluate(X_train, X_test, y_train, y_test)
    
    # 5. 可视化分析
    visualize_results(data, results, tfidf)

if __name__ == "__main__":
    main()
