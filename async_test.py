import asyncio
import time

# 这是一个异步函数，注意它前面多了一个 async 关键字
async def call_ai_model(task_id):
    print(f"任务 {task_id}: 发送问题给大模型...")
    
    # await 意思是：假装这里在等大模型思考 2 秒钟。
    # 它的魔法在于：等待的期间，电脑会自动去干别的活儿！
    await asyncio.sleep(2) 
    
    print(f"任务 {task_id}: 大模型回答完毕！")

# 奶茶店店长：同时处理三个任务
async def main():
    start_time = time.time()
    
    print("--- 奶茶店开始接单 ---")
    # asyncio.gather 的意思是：把这三个任务同时扔出去，一起等！
    await asyncio.gather(
        call_ai_model(1),
        call_ai_model(2),
        call_ai_model(3)
    )
    
    end_time = time.time()
    print(f"--- 全部完成，总共耗时：{end_time - start_time:.2f} 秒 ---")

# 启动异步程序（这是异步代码专属的启动方式）
asyncio.run(main())