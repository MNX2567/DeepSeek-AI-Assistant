# 1. 基础镜像：使用官方的 Python 环境
FROM python:3.10

# 2. 设置服务器内部的工作目录
WORKDIR /code

# 3. 复制依赖清单并安装
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 4. 复制所有代码到服务器中
COPY . .

# 5. 启动命令（注意：Hugging Face 规定必须监听 7860 端口）
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]