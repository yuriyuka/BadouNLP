import re


def extract_articles_from_civil_code(file_path):
    """
    从民法典文本中提取所有条款并保存为单独文件
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # 使用正则表达式匹配条款模式（第X条格式）
    pattern = r'(第[零一二三四五六七八九十百千条]+?)\s+(.*?)(?=第[零一二三四五六七八九十百千条]+|$)'
    articles = re.findall(pattern, content, re.DOTALL)

    # 处理每个条款
    for article_num, article_content in articles:
        # 清理内容中的多余空白字符
        article_content = article_content.strip()
        article_content = re.sub(r'\n+', '\n', article_content)
        article_content = re.sub(r' +', ' ', article_content)

        # 生成文件名
        filename = f"{article_num}.txt"

        # 写入文件
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"{article_num}\n\n")
            f.write(article_content)

        print(f"已生成: {filename}")

    print(f"\n共生成 {len(articles)} 个条款文件")


# 使用示例
if __name__ == "__main__":
    extract_articles_from_civil_code('民法典.txt')