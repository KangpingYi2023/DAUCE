#!/bin/bash
# 在 PG 容器内创建空 STATS 表和索引，实验脚本负责装载数据。
set -e
PSQL=/usr/local/pgsql/13.1/bin/psql
PG_PASSWORD=${PG_PASSWORD:-postgres}

ready=0
for attempt in $(seq 1 60); do
  if $PSQL -U postgres -d postgres -tAc "SELECT 1" >/dev/null 2>&1; then ready=1; break; fi
  sleep 1
done
[ "$ready" = 1 ] || { echo "PostgreSQL 启动超时" >&2; exit 1; }

$PSQL -v ON_ERROR_STOP=1 -U postgres -d postgres --set=password="$PG_PASSWORD" <<'SQL'
ALTER USER postgres PASSWORD :'password';
SQL
$PSQL -U postgres -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname='stats'" | grep -q 1 \
  || $PSQL -U postgres -d postgres -c "CREATE DATABASE stats;"
$PSQL -U postgres -d stats -tAc "SELECT 1 FROM pg_tables WHERE tablename='users'" | grep -q 1 \
  || $PSQL -v ON_ERROR_STOP=1 -U postgres -d stats -f /opt/stats/stats.sql
$PSQL -v ON_ERROR_STOP=1 -U postgres -d stats -f /opt/stats/indexes.sql
echo "stats 数据库就绪"
