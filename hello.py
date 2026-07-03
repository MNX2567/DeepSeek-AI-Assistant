print("Hello, AI World! 终于成功了！")
#变量Variables
# 在 Python 里，井号开头代表注释，这行字电脑不会执行，是写给人看的。
# 我们定义了三个变量：
user_name = "哲鑫"       # 这是一段文字（字符串）
target_salary = 20000    # 这是一个整数
is_ready = True          # 这是一个布尔值（真/假）

print(f"你好，{user_name}！你的目标薪资是 {target_salary}。")
#  lists
# 假设这是大模型支持的三种能力
ai_skills = ["写代码", "分析简历", "模拟面试"]

# 列表是从 0 开始数数的，取出第一个技能：
print("AI的第一个技能是：", ai_skills[0]) 

# 往列表里追加一个新技能：
ai_skills.append("翻译文档")

#Dictionaries(字典)
# 就像人的档案一样，左边是标签，右边是具体内容
ai_config = {
    "model_name": "DeepSeek-V3",
    "temperature": 0.7,
    "max_tokens": 2000
}

# 想要提取模型名字，直接报标签名即可：
print("当前使用的模型是：", ai_config["model_name"])

#函数Functions
# def 是 define(定义) 的缩写。这个函数负责生成给 AI 的提示词。
def build_prompt(role, task):
    # 这里的缩进（留白）非常重要，它代表这下面的代码属于这个函数
    prompt = f"你是一个专业的 {role}，请帮我完成这个任务：{task}"
    return prompt

# 调用这个工厂：
final_text = build_prompt("面试官", "对我的简历进行模拟提问")
print(final_text)