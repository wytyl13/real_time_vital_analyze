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










