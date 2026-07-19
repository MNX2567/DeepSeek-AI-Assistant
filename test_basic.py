import time

# 1. 这是一个我们自己写的“装饰器”（包装盒）
def time_logger(func):
    def wrapper():
        start_time = time.time()
        print("▶️ 开始执行函数...")
        
        func()  # 这里执行真正的函数
        
        end_time = time.time()
        print(f"⏹️ 函数执行完毕！耗时: {end_time - start_time:.4f} 秒")
    return wrapper

# 2. 我们用 @ 把包装盒套在普通函数上
@time_logger
def say_hello():
    print("你好，哲鑫！我是核心业务逻辑。")
    time.sleep(1) # 假装工作了 1 秒钟

# 3. 运行函数
say_hello()