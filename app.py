import os
import shutil
from flask import Flask, render_template, request, jsonify
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from zhipuai import ZhipuAI

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['CHROMA_FOLDER'] = 'chroma_db'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ====== 配置区：把你的API密钥填在这里 ======
ZHIPU_API_KEY = "你的智谱API密钥填这里"
# ===========================================

client = ZhipuAI(api_key=ZHIPU_API_KEY)


# 智谱 embedding 封装
class ZhipuEmbeddings(Embeddings):
    def __init__(self, api_key):
        self.client = ZhipuAI(api_key=api_key)

    def embed_documents(self, texts):
        response = self.client.embeddings.create(
            model="embedding-2",
            input=texts
        )
        return [item.embedding for item in response.data]

    def embed_query(self, text):
        response = self.client.embeddings.create(
            model="embedding-2",
            input=text
        )
        return response.data[0].embedding


embeddings = ZhipuEmbeddings(api_key=ZHIPU_API_KEY)
vector_store = None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    global vector_store
    try:
        if 'file' not in request.files:
            return jsonify({'error': '没有选择文件'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '文件名为空'}), 400

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        ext = os.path.splitext(file.filename)[1].lower()
        if ext == '.pdf':
            loader = PyPDFLoader(filepath)
        elif ext == '.docx':
            loader = Docx2txtLoader(filepath)
        elif ext in ['.md', '.txt']:
            loader = TextLoader(filepath, encoding='utf-8')
        else:
            return jsonify({'error': '不支持的文件格式，请上传PDF/Word/Markdown/TXT'}), 400

        documents = loader.load()
        if not documents:
            return jsonify({'error': '文档内容为空，请检查文件'}), 400

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "！", "？", ".", " "]
        )
        chunks = text_splitter.split_documents(documents)

        # 每次上传清空旧的向量数据库
        if os.path.exists(app.config['CHROMA_FOLDER']):
            shutil.rmtree(app.config['CHROMA_FOLDER'])

        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=app.config['CHROMA_FOLDER']
        )

        return jsonify({
            'message': f'文档上传成功！共切分为 {len(chunks)} 个文本块，可以开始提问了。'
        })

    except Exception as e:
        return jsonify({'error': f'上传失败：{str(e)}'}), 500


@app.route('/ask', methods=['POST'])
def ask():
    global vector_store
    try:
        if vector_store is None:
            return jsonify({'error': '请先上传文档'}), 400

        question = request.json.get('question', '')
        if not question:
            return jsonify({'error': '问题不能为空'}), 400

        # 直接用Chroma检索，不用LangChain的retriever，更稳定
        docs = vector_store.similarity_search(question, k=3)

        if not docs:
            return jsonify({'error': '未检索到相关内容，请换个问题试试'}), 400

        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""请根据以下文档内容回答用户的问题。如果文档中没有相关信息，请直接说"文档中未找到相关信息"，不要编造。

文档内容：
{context}

用户问题：{question}

请用中文回答："""

        response = client.chat.completions.create(
            model="glm-4-flash",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1000
        )

        answer = response.choices[0].message.content

        return jsonify({
            'answer': answer,
            'sources': [doc.page_content[:100] + "..." for doc in docs]
        })

    except Exception as e:
        return jsonify({'error': f'提问失败：{str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
