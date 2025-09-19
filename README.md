### real_time_vital_analyze
```

```


### 快速开始
sudo bash real_time_vital_analyze_docker_container_init.sh real_time_vital_analyze 6379 5433
docker network create real_time_vital_analyze
docker network connect real_time_vital_analyze real_time_vital_analyze_redis
docker network connect real_time_vital_analyze real_time_vital_analyze_postgres

git clone https://github.com/wytyl13/real_time_vital_analyze.git
conda create --name real_time_vital_analyze python=3.10
conda activate real_time_vital_analyze
cd real_time_vital_analyze
pip install -r requirements.txt
agent init







### api
```
[{"from": "replace_name", "to": "卫宇涛"}, {"from": "replace_gender", "to": "男"}, {"from": "replace_age", "to": "18"}, {"from": "replace_file_number", "to": "123456"}, {"from": "replace_description", "to": "我是谁？"}, {"from": "replace_evalutor", "to": "张三"}, {"from": "replace_evalution_doctor", "to": "李四"}, {"from": "replace_reviewer", "to": "王五"}, {"from": "replace_service_object", "to": "卫宇涛"}, {"from": "replace_service_object_family", "to": "李三"}, {"from": "replace_evalution_date", "to": "2025年9月2日"}]
```







