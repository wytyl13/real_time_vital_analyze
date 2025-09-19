
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import base64
import json
import urllib



def test_base64_onlyoffice(docx_file_path, server_url="https://ai.shunxikj.com:5002"):
    """
    测试OnlyOffice base64传参功能
    
    Args:
        docx_file_path: docx文件路径
        server_url: OnlyOffice服务器地址
    """
    
    # 1. 读取docx文件并转换为base64
    with open(docx_file_path, 'rb') as f:
        file_content = f.read()
    
    base64_data = base64.b64encode(file_content).decode('utf-8')
    
    # 2. 构建请求数据
    request_data = {
        "base64_data": base64_data,
        "base64_save": "http://localhost:5003/api/files/upload",
        "filename": "pinggu.docx",
        "user_id": "test_user_123",
        # "replace_information": json.dumps([
        #     {"from": "replace_name", "to": "舜熙科技算法"}, 
        #     {"from": "replace_gender", "to": "男"}, 
        #     {"from": "replace_age", "to": "18"}, 
        #     {"from": "replace_file_number", "to": "123456"}, 
        #     {"from": "replace_description", "to": "我是谁？"}, 
        #     {"from": "replace_evalutor", "to": "张三"}, 
        #     {"from": "replace_evalution_doctor", "to": "李四"}, 
        #     {"from": "replace_reviewer", "to": "王五"}, 
        #     {"from": "replace_service_object", "to": "舜熙科技算法"}, 
        #     {"from": "replace_service_object_family", "to": "李三"}, 
        #     {"from": "replace_evalution_date", "to": "2025年9月2日"}
        # ]),
        "extract_information": json.dumps([
            {"from": "B.1总计得分", "to": "elderlyAbilityEvaluationscore"},
            {"from": "B.2总计得分", "to": "exercisesAbilityEvaluationScore"},
            {"from": "B.3总计得分", "to": "mentalStateEvaluationScore"},
            {"from": "B.4总计得分", "to": "sensationPerceptionSocialParticipationscore"}
        ])
    }
    
    # 3. 发送POST请求
    response = requests.post(
        f"{server_url}/edit_url",
        json=request_data,
        headers={'Content-Type': 'application/json'},
        timeout=30
    )
    
    # 4. 检查结果
    if response.status_code == 200:
        print("测试成功！")
        print(f"响应类型: {response.headers.get('content-type')}")
        
        # 保存HTML响应到文件
        with open('onlyoffice_test.html', 'w', encoding='utf-8') as f:
            f.write(response.text)
        print("编辑器页面已保存到: onlyoffice_test.html")
    else:
        print(f"测试失败: {response.status_code}")
        print(f"错误信息: {response.text}")


def test_base64_onlyoffice_get(docx_file_path, server_url="https://localhost:5002"):
    """
    测试OnlyOffice base64传参功能 - GET请求版本
    
    Args:
        docx_file_path: docx文件路径
        server_url: OnlyOffice服务器地址
    """
    
    # 1. 读取docx文件并转换为base64
    with open(docx_file_path, 'rb') as f:
        file_content = f.read()
    
    base64_data = base64.b64encode(file_content).decode('utf-8')
    
    # 2. 构建URL参数
    replace_rules = [
        {"from": "{{用户名}}", "to": "张三"},
        {"from": "{{日期}}", "to": "2025-09-03"}
    ]
    
    params = {
        "base64_data": base64_data,
        "filename": "test_document.docx",
        "user_id": "test_user_123",
        "replace_information": json.dumps(replace_rules, ensure_ascii=False)
    }
    
    print(f"Base64数据长度: {len(base64_data)} 字符")
    print(f"完整URL长度: {len(f'{server_url}/edit_url') + len(urllib.parse.urlencode(params))} 字符")
    
    # 检查URL长度（通常浏览器限制在2048字符左右）
    full_url_length = len(f"{server_url}/edit_url?{urllib.parse.urlencode(params)}")
    if full_url_length > 2000:
        print(f"警告：URL长度 {full_url_length} 字符，可能超出浏览器限制")
    
    # 3. 发送GET请求
    try:
        response = requests.get(
            f"{server_url}/edit_url",
            params=params,
            timeout=30
        )
        
        # 4. 检查结果
        if response.status_code == 200:
            print("GET测试成功！")
            print(f"响应类型: {response.headers.get('content-type')}")
            
            # 保存HTML响应到文件
            with open('onlyoffice_test_get.html', 'w', encoding='utf-8') as f:
                f.write(response.text)
            print("编辑器页面已保存到: onlyoffice_test_get.html")
        else:
            print(f"GET测试失败: {response.status_code}")
            print(f"错误信息: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"GET请求异常: {e}")


def test_base64_upload_api(docx_file_path, server_url="https://ai.shunxikj.com:5002"):
    """
    测试base64文件上传API功能
    
    Args:
        docx_file_path: docx文件路径
        server_url: 上传API服务器地址
    """
    
    # 1. 读取docx文件并转换为base64
    with open(docx_file_path, 'rb') as f:
        file_content = f.read()
    
    base64_data = base64.b64encode(file_content).decode('utf-8')
    
    # 2. 构建请求数据（用于上传API）
    request_data = {
        "base64_data": base64_data,
        "filename": "test_upload_document.docx",
        "user_id": "test_user_123",
        "content_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    }
    
    print(f"开始测试base64上传...")
    print(f"原文件大小: {len(file_content)} bytes")
    print(f"base64编码长度: {len(base64_data)} 字符")
    print(f"用户ID: {request_data['user_id']}")
    print(f"文件名: {request_data['filename']}")
    
    # 3. 发送POST请求到上传API
    try:
        response = requests.post(
            f"{server_url}/api/files/upload",
            json=request_data,
            headers={'Content-Type': 'application/json'},
            timeout=60
        )
        
        # 4. 检查结果
        print(f"响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("上传成功！")
            print(f"响应数据: {json.dumps(result, indent=2, ensure_ascii=False)}")
            
            # 检查关键字段
            if result.get('success'):
                print("✓ 上传状态: 成功")
                print(f"✓ 保存路径: {result.get('file_path')}")
                print(f"✓ 文件名: {result.get('filename')}")
                print(f"✓ 用户ID: {result.get('user_id')}")
            else:
                print("✗ 上传失败")
                print(f"失败原因: {result.get('message')}")
                
        else:
            print(f"上传失败: {response.status_code}")
            try:
                error_data = response.json()
                print(f"错误信息: {json.dumps(error_data, indent=2, ensure_ascii=False)}")
            except:
                print(f"错误信息: {response.text}")
        
        return response
        
    except requests.exceptions.Timeout:
        print("请求超时！请检查服务器是否正常运行")
        return None
    except requests.exceptions.ConnectionError:
        print(f"连接错误！请检查服务器地址是否正确: {server_url}")
        return None
    except Exception as e:
        print(f"请求出错: {e}")
        return None



from agent.tool.retrieval import Retrieval

# 使用示例
if __name__ == "__main__":
    import asyncio
    
    # 替换为你的docx文件路径
    test_base64_onlyoffice("/work/ai/real_time_vital_analyze/api/source/111.docx")
    # test_base64_onlyoffice_get("/work/ai/real_time_vital_analyze/api/source/5.docx")

    # test_base64_upload_api("/work/ai/real_time_vital_analyze/api/source/5.docx")
#     text_list = [
#         {"123": """
#          A.5 健康相关问题
# A.5.1压力性损伤	•无
# •Ⅰ期：皮肤完好，出现指压不会变白的红印
# •Ⅱ期：皮肤真皮层损失、暴露，出现水泡
# •Ⅲ期：全层皮肤缺失，可见脂肪、肉芽组织以及边缘内卷
# •Ⅳ期：全层皮肤、组织缺失，可见肌腱、肌肉、腱膜，以及边缘内卷，伴随隧道、潜行
# •不可分期：全身皮肤、组织被腐肉、焦痂掩盖，无法确认组织缺失程度，去除腐肉、焦痂才可判断损伤程度
	
	
	
	
# A.5.2关节活动度	•无
# •是，影响日常生活功能，部位               
# •无法判断
	
# A.5.3伤口情况
# （可多选）	•无  •擦伤  •烧烫伤  •术后伤口  •糖尿病足溃疡  •血管性溃疡
# 	•其他伤口
# A.5.4特殊护理情况（可多选）	•无  •胃管  •尿管  •气管切开  •胃/肠/膀胱造瘘  •无创呼吸机
# 	•透析  •其他
# A.5.5疼痛感
# 注：通过表情反应和询问来判断	•无疼痛感  •轻度疼痛  •中度疼痛（尚可忍受的程度）
# 	•重度疼痛（无法忍受的程度）  •不知道或无法判断
# A.5.6牙齿缺失情
# 况（可多选）	•无缺失
# 	•牙体缺损（如龋齿、楔状缺损）
# 	•牙列缺损：O非对位牙缺失  O单侧对位牙缺失  O双侧对位牙缺失
# 	牙列缺失：O上颌牙缺失  O下颌牙缺失  O全口牙缺失
# A.5.7义齿佩戴情
# 况（可多选）	•无义齿  •固定义齿  •可摘局部义齿  •可摘全/半口义齿
# A.5.8吞咽困难的情形和症状
# （可多选）	•无
# 	•抱怨吞咽困难或吞咽时会疼痛
# 	•吃东西或喝水时出现咳嗽或呛咳
# 	•用餐后嘴中仍含着食物或留有残余食物
# 	•当喝或吃流质或者固体的食物时，食物会从嘴角边流失
# 	•有流口水的情况
# A.5.9营养不良：体质指数（BMI）低
# 于正常值
# 注:BMI=体重（kg）/【身高（m）】平方	•无   •有
	
	
# A.5.10清理呼吸道
# 无效	•无   •有
# A.5.11 昏迷	•无   •有
# A.5.12 其他（请补充）：


# B.1老年人能力评估表
# B.1.1  进食：使用适当的器具将食物送入口中并咽下
#    分	4分：独立使用器具将食物送进口中并咽下，没有呛咳
# 	3分：在他人指导或提示下完成，或独立使用辅具，没有呛咳
# 	2分：进食中至少需要少量接触式协助，偶尔（每月一次及以上）呛咳
# 	1分：在进食中需要大量接触式协助，经常（每周一次及以上）呛咳
# 	0分：完全依赖他人协助进食，或吞咽困难，或留置营养管
# B.1.2  修饰：指洗脸、刷牙、梳头、刮脸、剪指（趾）甲等
#    分	4分：独立完成，不需要协助
# 	3分：在他人指导或提示下完成
# 	2分：需要他人协助，但以自身完成为主
# 	1分：主要依靠他人协助，自身能给予配合
# 	0分：完全依赖他人协助，且不能给予配合
# B.1.3  洗澡：清洗和擦干身体
#    分	4分：独立完成，不需要协助
# 	3分：在他人指导或提示下完成
# 	2分：需要他人协助，但以自身完成为主
# 	1分：主要依靠他人协助，自身能给予配合
# 	0分：完全依赖他人协助，且不能给予配合
# B.1.4  穿/脱上衣：指穿/脱上身衣服、系扣、拉拉链等
#    分	4分：独立完成，不需要协助
# 	3分：在他人指导或提示下完成
# 	2分：需要他人协助，但以自身完成为主
# 	1分：主要依靠他人协助，自身能给予配合
# 	0分：完全依赖他人协助，且不能给予配合
# B.1.5  穿/脱裤子和鞋袜：指穿/脱裤子、鞋袜等
#    分	4分：独立完成，不需要协助
# 	3分：在他人指导或提示下完成
# 	2分：需要他人协助，但以自身完成为主
# 	1分：主要依靠他人协助，自身能给予配合
# 	0分：完全依赖他人协助，且不能给予配合
# B.1.6  小便控制：控制和排出尿液的能力
#    分	4分：可自行控制排尿，排尿次数、排尿控制均正常
# 	3分：白天可自行控制排尿次数，夜间出现排尿次数增多、排尿控制较差，或自行使用尿布、尿垫等辅助用物
# 	2分：白天大部分时间可自行控制排尿，偶出现（每天<1次，但每周>1次）尿失禁，夜间控制排尿较差，或他人少量协助使用尿布、尿垫等辅助用物
# 	1分：白天大部分时间不能控制排尿（每天≥1次，但尚非完全失控），夜间出现尿失禁，或他人大量协助使用尿布、尿垫等辅助用物
# 	0分：小便失禁，完全不能控制排尿，或留置导尿管
# B.1.7  大便控制：控制和排出粪便的能力
#    分	4分：可正常自行控制大便排出
# 	3分：有时出现（每天<1次）便秘或大便失禁，或自行使用开塞露、尿垫等辅助用物
# 	2分：经常出现（每天<1次，但每周>1次）便秘或大便失禁，或他人少量协助使用开塞露、尿垫等辅助用物
# 	1分：大部分时间均出现（每天≥1次）便秘或大便失禁，但尚非完全失控，或他人大量协助使用开塞露、尿垫等辅助用物
# 	0分：严重便秘或者完全大便失禁，需要依赖他人协助排便或清洁皮肤
# B.1.8  如厕：上厕所排泄大小便，并清洁身体
# 注：评估中强调排泄前解开裤子、完成排泄后清洁身体、穿上裤子
#    分	4分：独立完成，不需要协助
# 	3分：在他人指导或提示下完成
# 	2分：需要他人协助，但以自身完成为主
# 	1分：主要依靠他人协助，自身能给予配合
# 	0分：完全依赖他人协助，且不能给予配合
# B.1总计得分：



# B.2 基础运动能力评估表
# B.2.1  床上体位转移：卧床翻身及坐起躺下
#    分	4分：独立完成，不需要协助
# 	3分：在他人指导或提示下完成
# 	2分：需要他人协助，但以自身完成为主
# 	1分：主要依靠他人协助，自身能给予配合
# 	0分：完全依赖他人协助，且不能给予配合
# B.2.2  床椅转移：从坐位到站位，再从站位到坐位的转换过程
#    分	4分：独立完成，不需要协助
# 	3分：在他人指导或提示下完成
# 	2分：需要他人协助，但以自身完成为主
# 	1分：主要依靠他人协助，自身能给予配合
# 	0分：完全依赖他人协助，且不能给予配合
# B.2.3  平地行走：双脚交互的方式在地面行动，总是一只脚在前
#    注：包括他人辅助和使用辅具的步行
#    分	4分：独立平地行走50m左右，不需要协助，无摔倒风险
# 	3分：能平地行走50m左右，存在跌倒风险，需要他人监护或指导，或使用拐杖、助行器等辅助工具
# 	2分：在步行时需要他人少量扶持协助
# 	1分：在步行时需要他人大量扶持协助
# 	0分：完全不能步行
# B.2.4  上下楼梯：双脚交替完成楼梯台阶连续的上下移动
#    分	3分：可独立上下楼梯（连续上下10个~15个台阶），不需要协助
# 	2分：在他人指导或提示下完成
# 	1分：需要他人协助，但以自身完成为主
# 	0分：主要依靠他人协助，自身能给予配合；或者完全依赖他人协助，且不能给予配合
# B.2总计得分：





# B.3 精神状态评估表
# B.3.1  时间定向：知道并确认时间的能力
#    分	4分：时间观念（年、月）清楚，日期（或星期几）可相差一天
# 	3分：时间观念有些下降，年、月、日（或星期几）不能全部分清（相差两天或以上）
# 	2分：时间观念较差，年、月、日不清楚，可知上半年或下半年或季节
# 	1分：时间观念很差，年、月、日不清楚，可知上午、下午或白天、夜间
# 	0分：无时间观念
# B.3.2  空间定向：知道并确认空间的能力
#    分	4分：能在日常生活范围内单独外出，如在日常居住小区内独自外出购物等
# 	3分：不能单独外出，但能准确知道自己日常生活所在地的地址信息
# 	2分：不能单独外出，但知道较多有关自己日常生活的地址信息
# 	1分：不能单独外出，但知道较少自己居住或生活所在地的地址信息
# 	0分：不能单独外出，无空间观念
# B.3.3  人物定向：知道并确认人物的能力
#    分	4分：认识长期共同一起生活的人，能称呼并知道关系
# 	3分：能认识大部分共同生活居住的人，能称呼或知道关系
# 	2分：能认识部分日常同住的亲人或照护者等，能称呼或知道关系等
# 	1分：只认识自己或极少数日常同住的亲人或照护者等
# 	0分：不认识任何人（包括自己）
# B.3.4  记忆：短期近期和远期记忆能力
#    分	4分：总能保持与社会、年龄所适应的记忆能力，能完整的回忆
# 	3分：出现轻度的记忆紊乱或回忆不能（不能回忆即时信息3个词语经过5分钟后仅能回忆0~1个）
# 	2分：出现中度的记忆紊乱或回忆不能（不能回忆近期记忆，不记得上一顿饭吃了什么）
# 	1分：出现重度记忆紊乱或回忆不能（不能回忆远期记忆，不记得自己老朋友）
# 	0分：记忆完全紊乱或者完全不能对既往事物进行正确的回忆
# B.3.5  理解能力：理解语言信息和非语言信息的能力（可借助平时使用的助听设备等），即理解别
#        人的话
#    分	4分：能正常理解他人的话
# 	3分：能理解他人的话，但需要增加时间
# 	2分：理解有困难，需要频繁重复或简化口头表达
# 	1分：理解有严重困难，需要大量他人帮助
# 	0分：完全不能理解他人的话
# B.3.6  表达能力：表达信息能力，包括口头的和非口头的，即表达自己的想法
#    分	4分：能正常表达自己的想法
# 	3分：能表达自己的需要，但需要增加时间
# 	2分：表达需要有困难，需频繁重复或简化口头表达
# 	1分：表达有严重困难，需要大量他人帮助
# 	0分：完全不能表达需要
# B.3.7  攻击行为：身体攻击行为（如打/踢/推/咬/抓/摔东西）和语言攻击行为（如骂人、语言威胁、尖叫）
#    分	1分：未出现
# 	0分：近一个月内出现过攻击行为
# B.3.8  抑郁症状：存在情绪低落、兴趣减退、活力减退等症状，甚至出现妄想、幻觉、自杀念头或自杀行为  注：长期的负性情绪
#    分	1分：未出现
# 	0分：近一个月内出现过负性情绪
# B.3.9  意识水平：机体对自身和周围环境的刺激做出应答反应的能力程度，包括清醒和持续的觉醒状态  注：处于昏迷状态者，直接评定为重度失能
#     分	2分：神志清醒，对周围环境能做出正确反映
# 	1分：嗜睡，表现为睡眠状态过度延长。当呼唤或推动老年人的肢体时可唤醒，并能进行正确的交谈或执行指令，停止刺激后又继续入睡；意识模糊，注意力涣散，对外界刺激不能清晰的认识，空间和时间定向力障碍，理解力迟钝，记忆力模糊和不连贯
# 	0分：昏睡，一般的外界刺激不能使其觉醒，给予较强烈的刺激时可有短时的意识清醒，醒后可简短回答提问，当刺激减弱后又很快进入睡眠状态；或者昏迷；意识丧失，随意运动丧失，对一般刺激全无反应
# B.3总计得分：




# B.4 感知觉与社会参与评估表
# B.4.1  视力：感受存在的光线并感受物体的大小，形状的能力。在个体的最好矫正视力下进行评估
#     分	2分：视力正常
# 	1分：能看清楚大字体，但看不清书报上的标准字体；视力有限，看不清楚报纸大标题，但能辨认物体
# 	0分：只能看到光、颜色和形状；完全失明
# B.4.2  听力：能辨别声音的方位、音调、音量和音质的有关能力（可借助平时使用助听设备等）
#    分	2分：听力正常
# 	1分：在轻声说话或说话距离超过两米时听不清；正常交流有些困难，需要在安静的环境或大声说话才能听到
# 	0分：讲话者大声说话或说话很慢，才能部分听见；完全失聪
# B.4.3  执行日常事务：计划、安排并完成日常事务，包括但不限于洗衣服、小金额购物、服药管理
#    分	4分：能完全独立计划、安排和完成日常事务，无需协助
# 	3分：在计划、安排和完成日常事务是需要他人监护或指导
# 	2分：在计划、安排和完成日常事务是需要少量协助
# 	1分：在计划、安排和完成日常事务是需要大量协助
# 	0分：完全依赖他人进行日常事务
# B.4.4  使用交通工具外出
#    分	3分：能自己骑车或搭乘公共交通工具外出
# 	2分：能自己搭乘出租车，但不会搭乘其他交通工具外出
# 	1分：当有人协助或陪伴，可搭乘公共交通工具外出
# 	0分：只能在他人协助下搭乘出租车或私家车外出；完全不能出门，或者外出完全需要协助
# B.4.5  社会交往能力
#    分	4分：参与社会，在社会环境有一定的适应能力，待人接物恰当
# 	3分：能适应单纯环境，主动接触他人，初次见面时难让人发现智力问题，不能理解隐喻语
# 	2分：脱离社会，可被动接触，不会主动待他人，谈话中很多不适词句，容易上当受骗
# 	1分：勉强可与他人接触，谈吐内容不清楚，表情不恰当
# 	0分：不能与人交往
# B.4总计得分：


#          """}, 
#     ]
#     # text_list = []
#     retrieval = Retrieval(
#         # data_dir="/work/ai/community_agent/retrieval_data", 
#         # index_dir="/work/ai/community_agent/retrieval_storage", 
#         chunk_size=32, 
#         chunk_overlap=20, 
#         line_based_chunk=False
#     )
    
    # retrieval_words = [
    #     {"from": "B.1总计得分", "to": "elderlyAbilityEvaluationscore"},
    #     {"from": "B.2总计得分", "to": "exercisesAbilityEvaluationScore"},
    #     {"from": "B.3总计得分", "to": "mentalStateEvaluationScore"},
    #     {"from": "B.4总计得分", "to": "sensationPerceptionSocialParticipationscore"}
    # ]
    
    # retrieval_words = "".join([item["from"] for item in retrieval_words])
    # async def main():
    #     nodes = await retrieval.execute(text_list=text_list, retrieval_word='2025年9月1日时讯消息', top_k=3)

    #     print([node.text for node in nodes])
    # asyncio.run(main())
    # retrieval.add_text(
    #     text="2025年7月20日 早上7点 物业门口早餐菜品有：豆腐脑、咸菜",
    #     text_id="2"
    # )
    # retrieval.show_nodes()
    # retrieval.delete_text(
    #     text_id="1"
    # )
    # retrieval.show_nodes()
    # async def main():
    #     result = await retrieval.execute(text_list=text_list, retrieval_word=retrieval_words, top_k=1)
    #     print(result)
    # asyncio.run(main())
    
    
    
    
    
    
    
    