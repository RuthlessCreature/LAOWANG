# DB 初始化与排错（MySQL）

本项目默认只推荐使用本地 MySQL（配置见 `config.ini`），核心表结构在 `sql/schema_mysql.sql`。

## 1) 启动 MySQL（两种方式）

### 方式 A：Docker（仓库提供 compose）

前提：已安装 Docker Desktop + `docker compose`。

```bash
docker compose -f docker-compose.mysql.yml up -d
```

注意：`docker-compose.mysql.yml` 内置了示例账号/库名（如 `astock/astock_pass` / `astock`）。
请同步修改 `config.ini` 里的 `db_url` 或 `[mysql]` 配置。

### 方式 B：本机安装 MySQL

确保你已经创建好数据库（如 `astock`）并准备好账号权限：

```sql
CREATE DATABASE astock DEFAULT CHARSET utf8mb4;
GRANT ALL PRIVILEGES ON astock.* TO 'astock'@'%' IDENTIFIED BY '你的密码';
FLUSH PRIVILEGES;
```

## 2) 配置 `config.ini`

推荐直接写：

- `[database] db_url = mysql+pymysql://user:password@127.0.0.1:3306/astock?charset=utf8mb4`

也可以填 `[mysql]` 段让程序自动拼接。

连接优先级（高 → 低）：

1. CLI `--db-url`
2. 环境变量 `ASTOCK_DB_URL`
3. CLI `--db`（SQLite 文件路径）
4. `config.ini`
5. 默认 SQLite：`data/stock.db`（不推荐）

## 3) 初始化表结构（只需一次）

推荐（脚本方式；要求数据库已存在）：

- `python init.py --config config.ini`

等价命令：

```bash
python astock_analyzer.py --config config.ini init-db
python models_update.py --config config.ini init-tables
```

如果你希望用纯 SQL 一次性建库建表：

```bash
mysql -u astock -p -h 127.0.0.1 -P 3306 < sql/schema_mysql.sql
```

（该 SQL 文件会创建数据库 `astock`；如需别的库名请自行改 `schema_mysql.sql` 的 `CREATE DATABASE/USE`）

## 4) 一键诊断（推荐）

```powershell
scripts/db_doctor.ps1
```

它会输出：
- 实际解析到的 DB（会对密码打码）
- 方言（mysql/sqlite）
- 当前库里的表数量
- 核心表是否齐全（`stock_daily` / `stock_scores_v3` / `model_*` 等）

## 5) 常见问题

### 5.1 明明配了 MySQL，却跑到 SQLite

99% 是被 `ASTOCK_DB_URL` 或 CLI `--db data/stock.db` 覆盖了。
建议：
- 日常不要长期设置 `ASTOCK_DB_URL`
- 只用 `config.ini` 管理连接

### 5.2 `no such table: stock_daily`

含义：连到了空库（MySQL 选错库 / 没初始化 / 或误连 SQLite 文件）。
处理：
1) `scripts/db_doctor.ps1` 看最终连到哪里、核心表是否存在  
2) 再跑 `python init.py --config config.ini` 或 `python astock_analyzer.py --config config.ini init-db`
